"""Local confined Docker execution with durable per-run effect ownership."""
import fcntl,json,os,re,subprocess,threading,time,uuid
from pathlib import Path
from code_scoring import CodeScoringFailure
from evalscope.api.mixin.code_execution_sandbox_mixin import CodeExecutionBackend
_REMOTE_EXECUTION_LOCK=threading.Lock()
CANCELLED=threading.Event()
_EXECUTIONS=threading.Condition()
_INFLIGHT={}

def cancel_and_drain(timeout=60):
    CANCELLED.set()
    deadline=time.monotonic()+timeout
    with _EXECUTIONS:
        while _INFLIGHT:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise CleanupUnknown('In-flight executions retain durable ownership',{'inflight_work_dirs':list(_INFLIGHT.values())})
            _EXECUTIONS.wait(min(.1,remaining))
    return {'state':'drained','inflight':0}


class ExecutionInfrastructure(CodeScoringFailure):
    kind='execution_infrastructure'
    def __init__(self,detail,metadata=None):
        self.metadata={} if metadata is None else metadata
        super().__init__(detail+'; ownership='+str(self.metadata.get('ownership_path','unset_before_create')))
class OwnershipJournalRequired(ExecutionInfrastructure):kind='ownership_journal_required'
class OwnershipPersistenceFailed(ExecutionInfrastructure):kind='ownership_persistence_failed'
class CreationOutcomeUnknown(ExecutionInfrastructure):kind='create_outcome_unknown'
class CleanupUnknown(ExecutionInfrastructure):kind='cleanup_unknown'

def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def durable_directory(path):
    missing=[];cursor=path
    while not cursor.exists():missing.append(cursor);cursor=cursor.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True);fsync_directory(directory.parent)

class OwnershipJournal:
    def __init__(self,task_config):
        value=getattr(task_config,'work_dir',None)
        if not isinstance(value,(str,Path)) or not str(value).strip() or not Path(value).is_absolute():
            raise OwnershipJournalRequired('Code execution requires an explicit absolute TaskConfig.work_dir')
        self.path=Path(value)/'execution-journal.jsonl'
        self.active=self.path.parent/'active-executions'
    def persist(self,metadata):
        path=Path(metadata['ownership_path']);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            durable_directory(self.active)
            payload=(json.dumps(metadata,sort_keys=True)+'\n').encode()
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,path);fsync_directory(path.parent)
            created=not self.path.exists()
            with self.path.open('ab') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX);stream.write(payload);stream.flush();os.fsync(stream.fileno())
            if created:fsync_directory(self.path.parent)
        except OSError as error:raise OwnershipPersistenceFailed(type(error).__name__+': '+str(error),metadata) from error
    def reserve(self,image):
        name='bonsai-quality-'+uuid.uuid4().hex
        metadata={'image':image,'network':'none','host_mounts':[],'container_name':name,'container_id':None,
            'phase':'create_intent','creation_outcome':'not_submitted','cleanup_outcome':'not_required',
            'journal_path':str(self.path),'ownership_path':str(self.active/(name+'.json'))}
        self.persist(metadata)
        return metadata
    def release(self,metadata):
        self.persist(metadata)
        try:Path(metadata['ownership_path']).unlink();fsync_directory(self.active)
        except OSError as error:raise OwnershipPersistenceFailed(str(error),metadata) from error

def observed_program(code):
    marker='bonsai-tests-completed-'+uuid.uuid4().hex
    program='import sys\ndef _observe():\n emit=sys.stdout.write\n exec(compile('+repr(code)+', "<stdin>", "exec"), {"__name__":"__main__", "__file__":"<stdin>"})\n emit('+repr('\n'+marker+'\n')+')\n_observe()\n'
    return program,marker

def observe_result(result,marker):
    if result['status']=='success':
        if marker not in result['output'].splitlines():
            result.update(status='error',error='Code exited before the complete test program returned')
            result['metadata']['premature_exit']=True
        else:
            result['output']=result['output'].replace('\n'+marker+'\n','')
            result['metadata']['tests_completed']=True
    return result

def absence(result):return result.returncode!=0 and any(s in result.stderr.lower() for s in ['no such object','no such container'])

class PinnedDockerBackend(CodeExecutionBackend):
    def __init__(self,benchmark_meta,task_config,image):
        super().__init__(benchmark_meta,task_config);self.image=image;self.ready=False;self.journal=OwnershipJournal(task_config)
    def start(self):
        if not self.image.startswith('sha256:'):raise ExecutionInfrastructure('Sandbox must use immutable Docker image ID')
        try:subprocess.run(['docker','image','inspect',self.image],check=True,capture_output=True,timeout=15)
        except (OSError,subprocess.SubprocessError) as error:raise ExecutionInfrastructure(str(error)) from error
        self.ready=True
    def is_ready(self):return self.ready
    def stop(self):self.ready=False
    def reconcile_creation(self,docker,metadata):
        metadata.update(phase='create_reconciliation',creation_outcome='unknown',cleanup_outcome='pending_reconciliation');self.journal.persist(metadata)
        deadline=time.monotonic()+20
        try:
            while time.monotonic()<deadline:
                response=subprocess.run(docker+['inspect','--format','{{.Id}}',metadata['container_name']],capture_output=True,text=True,timeout=3)
                if response.returncode==0:
                    identity=response.stdout.strip()
                    if not re.fullmatch('[0-9a-f]{64}',identity):raise ValueError('Invalid reconciled container ID')
                    metadata.update(container_id=identity,creation_outcome='reconciled_after_uncertain_create');self.journal.persist(metadata);return
                metadata['last_create_inspection']={'returncode':response.returncode,'stderr':response.stderr}
                time.sleep(.1)
        except (OSError,subprocess.SubprocessError,ValueError,KeyboardInterrupt) as error:
            metadata['reconciliation_error']={'type':type(error).__name__,'detail':str(error)}
        metadata.update(phase='create_outcome_unknown',cleanup_outcome='unknown_create_retained');self.journal.persist(metadata)
        raise CreationOutcomeUnknown('Create may still commit; negative early inspections do not establish absence',metadata)
    def cleanup(self,docker,metadata):
        identity=metadata['container_id']
        if identity is None:raise CreationOutcomeUnknown('No confirmed container ID for cleanup',metadata)
        metadata.update(phase='cleanup_intent',cleanup_outcome='pending');self.journal.persist(metadata)
        try:
            removed=subprocess.run(docker+['rm','-f',identity],capture_output=True,text=True,timeout=15)
            metadata.update(cleanup_returncode=removed.returncode,cleanup_stdout=removed.stdout,cleanup_stderr=removed.stderr)
            self.journal.persist(metadata)
            inspected=subprocess.run(docker+['inspect',identity],capture_output=True,text=True,timeout=5)
            metadata['cleanup_inspection']={'returncode':inspected.returncode,'stdout':inspected.stdout,'stderr':inspected.stderr}
            if not absence(inspected) or (removed.returncode!=0 and not absence(removed)):raise ValueError('Removal/absence not confirmed')
        except (OSError,subprocess.SubprocessError,ValueError,KeyboardInterrupt) as error:
            metadata.update(phase='cleanup_unknown',cleanup_outcome='unknown',cleanup_error={'type':type(error).__name__,'detail':str(error)})
            self.journal.persist(metadata);raise CleanupUnknown('Owned Docker cleanup not confirmed',metadata) from error
        metadata.update(phase='closed',cleanup_outcome='confirmed_absent');self.journal.release(metadata)
    def execute(self,code,timeout,language):
        token=uuid.uuid4().hex
        with _EXECUTIONS:
            if CANCELLED.is_set():raise ExecutionInfrastructure('Execution cancelled before a new Docker create')
            _INFLIGHT[token]=str(self.journal.path.parent)
        try:return self._execute(code,timeout,language)
        finally:
            with _EXECUTIONS:
                del _INFLIGHT[token];_EXECUTIONS.notify_all()
    def _execute(self,code,timeout,language):
        if language.lower()!='python':raise ExecutionInfrastructure('Pinned local sandbox supports Python scoring only')
        if not isinstance(timeout,(int,float)) or isinstance(timeout,bool) or timeout<=0:raise ExecutionInfrastructure('Positive program timeout required')
        if isinstance(code,list):code='\n'.join(code)
        code,marker=observed_program(code)
        metadata=self.journal.reserve(self.image)
        docker=['docker']
        command=docker+['create','--name',metadata['container_name'],'--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','128','--memory','2g','--cpus','2',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=256m','--user','65534:65534','--workdir','/tmp',
            '-e','HTTP_PROXY=','-e','HTTPS_PROXY=','-e','ALL_PROXY=','-e','NO_PROXY=',
            '-i',self.image,'python','-I','-']
        create_attempted=False
        try:
            metadata.update(phase='create_submitted',creation_outcome='pending');self.journal.persist(metadata)
            create_attempted=True
            creation=subprocess.run(command,text=True,capture_output=True,timeout=15)
            metadata.update(create_returncode=creation.returncode,create_stdout=creation.stdout.strip(),create_stderr=creation.stderr)
            if creation.returncode!=0 or not re.fullmatch('[0-9a-f]{64}',creation.stdout.strip()):raise ExecutionInfrastructure('Container create did not return a successful exact identity',metadata)
            metadata.update(container_id=creation.stdout.strip(),creation_outcome='confirmed',phase='program_start');self.journal.persist(metadata)
            timed_out=False
            try:program=subprocess.run(docker+['start','--attach','--interactive',metadata['container_id']],input=code,text=True,capture_output=True,timeout=timeout)
            except subprocess.TimeoutExpired:timed_out=True
            inspection=subprocess.run(docker+['inspect','--format','{{json .State}}',metadata['container_id']],capture_output=True,text=True,timeout=15)
            if inspection.returncode:raise ExecutionInfrastructure('Owned container state unavailable: '+inspection.stderr,metadata)
            state=json.loads(inspection.stdout);metadata['container_state']=state
            if state['StartedAt'].startswith('0001-') or state['Error']:raise ExecutionInfrastructure('Container program did not start',metadata)
            if timed_out:return {'status':'timeout','error':'Official scorer execution exceeded timeout','output':'','metadata':metadata}
            if state['Running'] or state['Status']!='exited' or state['ExitCode']!=program.returncode:raise ExecutionInfrastructure('Attachment did not observe the program exit',metadata)
            metadata.update(exit_code=program.returncode,attachment_returncode=program.returncode)
            return observe_result({'status':'success' if program.returncode==0 else 'error','output':program.stdout[-1048576:],
                'error':program.stderr[-1048576:] if program.returncode else None,'metadata':metadata},marker)
        except (OSError,subprocess.SubprocessError,ValueError,KeyError,KeyboardInterrupt) as error:
            metadata['execution_error']={'type':type(error).__name__,'detail':str(error)};self.journal.persist(metadata)
            raise ExecutionInfrastructure('Docker execution failed: '+str(error),metadata) from error
        finally:
            if create_attempted:
                if metadata['container_id'] is None:self.reconcile_creation(docker,metadata)
                self.cleanup(docker,metadata)

def install_backend(image):
    """Every local backend derives mandatory ownership custody from explicit run work_dir."""
    from evalscope.api.mixin.code_execution_sandbox_mixin import CodeExecutionSandboxMixin
    def factory(self):
        if self._backend is None:self._backend=PinnedDockerBackend(self._benchmark_meta,self._task_config,image)
        return self._backend
    CodeExecutionSandboxMixin._get_backend=factory
    from code_scoring import install
    install()

class RemotePinnedDockerBackend(PinnedDockerBackend):
    """Same execution contract via the CPU-only grader VM's loopback-forwarded API."""
    def __init__(self,benchmark_meta,task_config,image,endpoint):
        CodeExecutionBackend.__init__(self,benchmark_meta,task_config);self.image=image;self.ready=False;self.endpoint=endpoint.rstrip('/')
    def start(self):
        import urllib.request
        with urllib.request.urlopen(self.endpoint+'/health',timeout=30) as response:health=json.load(response)
        if self.image not in health.get('images',[]):raise ValueError('Requested pinned grader image absent from VM health')
        self.ready=True
    def execute(self,code,timeout,language):
        import urllib.request
        if isinstance(code,list):code='\n'.join(code)
        code,marker=observed_program(code)
        request={'code':code,'timeout':timeout,'language':language,'image':self.image}
        req=urllib.request.Request(self.endpoint+'/execute',json.dumps(request).encode(),{'Content-Type':'application/json'})
        import urllib.error
        with _REMOTE_EXECUTION_LOCK:
            deadline=time.monotonic()+timeout+120
            while True:
                try:
                    with urllib.request.urlopen(req,timeout=timeout+120) as response:result=json.load(response)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code!=503 or time.monotonic()>=deadline:raise
                    time.sleep(1)
        if result.get('status') not in ['success','error','timeout']:raise ValueError('Grader VM returned invalid execution status')
        return observe_result(result,marker)

def install_remote_backend(image,endpoint):
    from evalscope.api.mixin.code_execution_sandbox_mixin import CodeExecutionSandboxMixin
    def factory(self):
        if self._backend is None:self._backend=RemotePinnedDockerBackend(self._benchmark_meta,self._task_config,image,endpoint)
        return self._backend
    CodeExecutionSandboxMixin._get_backend=factory
    from code_scoring import install
    install()
