#!/usr/bin/env python3
"""Audit every function for ring releases with pending warp-local shared loads.

Shared-memory loads complete in order. Waiting on a load's scoreboard (or that of
any later shared load) completes it. An earlier empty SYNCS.ARRIVE.A1T0 can let the
producer overwrite those reads. This is a linear instruction-order audit, not CFG
analysis. Unconditional branches reset the pending set as before.

Usage: sass_release_check.py FILE [--all] [--sass]
FILE is a cubin/binary by default (cuobjdump must be on PATH); --sass reads a saved
cuobjdump -sass dump. Exit 0: clean, 1: findings, 2: input/tool/format error.
Parser measurements and contract: tools/sass_release_check.md.
"""
import argparse
import re
import subprocess
import sys
import tempfile

ADDRESS = re.compile(r'[0-9a-f]{4,}')
WORD = re.compile(r'0x[0-9a-f]+')
SHARED_LOAD = re.compile(r'(LDS|LDSM)\b|LDS\.|LDSM\.')


class SassFormatError(ValueError):
    pass


def decode(hi):
    c = hi >> 41
    return (c >> 5) & 7, (c >> 11) & 0x3f


def encoding(text):
    text = text.strip()
    if not text.startswith('/*') or not text.endswith('*/'):
        raise SassFormatError('missing instruction encoding comment')
    word = text[2:-2].strip()
    if not WORD.fullmatch(word):
        raise SassFormatError(f'invalid instruction encoding: {word!r}')
    return int(word, 16)


def parse_functions(lines):
    name, body = None, []
    lines = enumerate(lines, 1)
    for number, line in lines:
        text = line.lstrip()
        if text.startswith('Function : '):
            if name is not None:
                if not body:
                    raise SassFormatError(f'{name}: no decoded instructions')
                yield name, body
            name = text[len('Function : '):].strip()
            if not name or len(name.split()) != 1:
                raise SassFormatError(f'line {number}: invalid function name')
            body = []
            continue
        if not text.startswith('/*'):
            continue
        address, end, rest = text[2:].partition('*/')
        if not end or not ADDRESS.fullmatch(address):
            continue
        if name is None:
            raise SassFormatError(f'line {number}: instruction outside a function')
        try:
            # Delimiters, not overlapping whitespace regexes: cuobjdump pads wide
            # operand/annotation columns, making the old matcher backtrack cubically.
            ins, comment, word = rest.rpartition('/*')
            if not comment:
                raise SassFormatError('missing low instruction word')
            encoding(comment + word)
            ins = ins.strip().removesuffix(';').rstrip()
            if not ins or (ins.startswith('@') and len(ins.split()) < 2):
                raise SassFormatError('missing instruction opcode')
            high = next(lines, None)
            if high is None:
                raise SassFormatError('missing high instruction word')
            wb, wait = decode(encoding(high[1]))
        except SassFormatError as error:
            raise SassFormatError(f'{name}, line {number}: {error}') from error
        body.append((int(address, 16), ins, wb, wait))
    if name is None:
        raise SassFormatError('no SASS functions found')
    if not body:
        raise SassFormatError(f'{name}: no decoded instructions')
    yield name, body


def functions(path):
    with tempfile.TemporaryFile(mode='w+t') as errors:
        process = subprocess.Popen(['cuobjdump', '-sass', path], stdout=subprocess.PIPE,
                                   stderr=errors, text=True)
        try:
            yield from parse_functions(process.stdout)
            code = process.wait()
            if code:
                errors.seek(0)
                raise subprocess.CalledProcessError(code, process.args, stderr=errors.read(65536))
        except SassFormatError:
            # A failed disassembler may produce no/partial SASS. Report its actual
            # tool error rather than claiming the binary itself has a format defect.
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            code = process.wait()
            errors.seek(0)
            diagnostic = errors.read(65536)
            if code and diagnostic:
                raise subprocess.CalledProcessError(code, process.args, stderr=diagnostic)
            raise
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


def check(body):
    findings = []
    pending = []
    for addr, ins, wb, wait in body:
        if wait and pending:
            last = -1
            for k, (_, _, b) in enumerate(pending):
                if b != 7 and (wait >> b) & 1:
                    last = k
            if last >= 0:
                pending = pending[last + 1:]
        if ins.startswith('@!PT '):
            continue
        op = ins.split()[0] if not ins.startswith('@') else ins.split()[1]
        if op.startswith(('MEMBAR', 'FENCE.VIEW.ASYNC')):
            pending = []
            continue
        if SHARED_LOAD.match(op):
            pending.append((addr, ins, wb))
        elif op.startswith('SYNCS.ARRIVE') and 'A1T0' in op and pending:
            findings.append((addr, ins, list(pending)))
        if op.startswith(('BRA', 'EXIT', 'BAR', 'RET')) and not ins.startswith('@'):
            pending = []
    return findings


def audit(stream, show_all):
    bad = count = instructions = 0
    for name, body in stream:
        count += 1
        instructions += len(body)
        findings = check(body)
        if findings:
            bad += 1
            print(f'{name}: {len(findings)} release(s) with pending shared loads', flush=True)
            if show_all:
                for addr, ins, pending in findings:
                    print(f'  {addr:05x} {ins}')
                    for a, text, _ in pending:
                        print(f'      pending {a:05x} {text}')
    print(f'{count} function(s) audited, {instructions} instruction(s)')
    print(f'{bad} function(s) affected')
    return 1 if bad else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--all', action='store_true', help='show releases and their pending loads')
    parser.add_argument('--sass', action='store_true', help='read saved cuobjdump -sass text')
    args = parser.parse_args(argv)
    try:
        if args.sass:
            with open(args.path) as source:
                return audit(parse_functions(source), args.all)
        return audit(functions(args.path), args.all)
    except (OSError, SassFormatError, subprocess.CalledProcessError) as error:
        print(f'SASS audit error: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
