#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('path', type=Path)
    args=ap.parse_args()
    decoder=json.JSONDecoder()
    marker='"traces":['
    prefix=''
    with args.path.open(encoding='utf-8') as fh:
        while marker not in prefix:
            chunk=fh.read(1<<20)
            if not chunk:
                raise SystemExit('top-level traces array not found')
            prefix+=chunk
        before, buffer=prefix.split(marker,1)
        metric_marker='"node_token_usage_samples":'
        metric_pos=before.index(metric_marker)+len(metric_marker)
        node_samples, _=decoder.raw_decode(before, metric_pos)
        declared=int(re.search(r'"trace_count":(\d+)', before).group(1))
        count=requests=bad=0
        missing_prefill=missing_decode=missing_cache=missing_prefill_latency=missing_decode_latency=0
        while True:
            buffer=buffer.lstrip()
            if buffer.startswith(','):
                buffer=buffer[1:].lstrip()
            if buffer.startswith(']'):
                break
            try:
                item,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk=fh.read(1<<20)
                if not chunk:
                    raise
                buffer+=chunk
                continue
            count+=1
            traces=item.get('trajectory',{}).get('traces',[])
            valid=bool(item.get('session_id')) and bool(traces) and all(
                isinstance(t.get('prompt_messages'),list) and t.get('prompt_messages') and
                isinstance(t.get('response_messages'),list) and t.get('response_messages')
                for t in traces if isinstance(t,dict)
            ) and all(isinstance(t,dict) for t in traces)
            if not valid:
                bad+=1
            for req in item.get('requests',[]):
                requests+=1
                if not req.get('request_id'):
                    bad+=1
                missing_prefill+=req.get('prefill_token_count') is None
                missing_decode+=req.get('decode_token_count') is None
                missing_cache+=req.get('cache_token_count') is None
                missing_prefill_latency+=req.get('prefill_latency_ms') is None
                missing_decode_latency+=req.get('decode_latency_ms') is None
            buffer=buffer[end:]
    digest=hashlib.sha256()
    with args.path.open('rb') as fh:
        for chunk in iter(lambda:fh.read(8<<20),b''):
            digest.update(chunk)
    result={
        'declared_trace_count':declared,
        'parsed_trace_count':count,
        'invalid_selected_traces':bad,
        'request_count':requests,
        'node_metric_samples':len(node_samples),
        'nodes':sorted({s.get('node') for s in node_samples if s.get('node')}),
        'request_missing_counts':{
            'prefill_token_count':missing_prefill,
            'decode_token_count':missing_decode,
            'cache_token_count':missing_cache,
            'prefill_latency_ms':missing_prefill_latency,
            'decode_latency_ms':missing_decode_latency,
        },
        'bytes':args.path.stat().st_size,
        'sha256':digest.hexdigest(),
    }
    print(json.dumps(result,sort_keys=True))
    if declared!=300 or count!=300 or bad:
        raise SystemExit(2)

if __name__=='__main__':
    main()
