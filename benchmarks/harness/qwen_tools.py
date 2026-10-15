"""CPU-only execution of the pinned Qwen/vLLM non-streaming tool parser.

The exact Apache-2.0 reference bytes and hash are assets/tool-parser-source.json.
Only non-streaming methods are compiled; CUDA/vLLM serving imports are unnecessary.
This is a campaign parser pin; the publisher did not disclose their parser commit.
"""
import ast,hashlib,json,logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any,List,Optional
import regex as re
ROOT=Path(__file__).resolve().parent.parent/'assets'
METHODS={'_get_arguments_config','_convert_param_value','_parse_xml_function_call','_get_function_calls','extract_tool_calls'}

def parse(final,tools):
 source=(ROOT/'qwen3coder_tool_parser.reference.py').read_bytes();pin=json.loads((ROOT/'tool-parser-source.json').read_text())
 if hashlib.sha256(source).hexdigest()!=pin['sha256']:raise ValueError('Pinned tool parser source hash changed')
 cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='Qwen3CoderToolParser')
 cls.bases=[];cls.decorator_list=[];cls.body=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in METHODS]
 for method in cls.body:
  method.returns=None
  for arg in method.args.args:arg.annotation=None
 scope={'ast':ast,'json':json,'re':re,'logger':logging.getLogger('pinned_qwen_tools'),'ToolCall':SimpleNamespace,'FunctionCall':SimpleNamespace,'ExtractedToolCallInformation':SimpleNamespace}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'<pinned-qwen-tools>','exec'),scope)
 parser=scope['Qwen3CoderToolParser']();parser.tool_call_prefix='<function=';parser.tool_call_start_token='<tool_call>';parser.prev_tool_call_arr=[]
 parser.tool_call_regex=re.compile(r'<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$',re.DOTALL)
 parser.tool_call_function_regex=re.compile(r'<function=(.*?)</function>|<function=(.*)$',re.DOTALL)
 parser.tool_call_parameter_regex=re.compile(r'<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)',re.DOTALL)
 schemas=[SimpleNamespace(type=t['type'],function=SimpleNamespace(**t['function'])) for t in tools]
 parsed=parser.extract_tool_calls(final,SimpleNamespace(tools=schemas));calls=[]
 for index,call in enumerate(parsed.tool_calls):
  calls.append({'id':'call_'+hashlib.sha256((final+str(index)).encode()).hexdigest()[:24],'type':'function','function':{'name':call.function.name,'arguments':call.function.arguments}})
 return parsed.content,calls,pin

def normalize_history(messages):
 result=[]
 for message in messages:
  message=dict(message)
  if message.get('tool_calls'):
   calls=[]
   for tool in message['tool_calls']:
    tool=dict(tool);tool['function']=dict(tool['function']);value=tool['function'].get('arguments',{})
    if isinstance(value,str):tool['function']['arguments']=json.loads(value)
    calls.append(tool)
   message['tool_calls']=calls
  result.append(message)
 return result
