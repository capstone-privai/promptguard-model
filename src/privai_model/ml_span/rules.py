"""Reuse baseline scanner initialization/log privacy, expose full-text value spans."""
import hashlib
import json
from pathlib import Path
from privai_model.baselines.credsweeper import CredSweeperBaseline,private_scanner_logs
from .schema import validate_spans


class RuleSpanDetector:
    def __init__(self):
        self.baseline=CredSweeperBaseline(use_ml=False)
        import credsweeper
        root=Path(credsweeper.__file__).parent
        hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file() and p.suffix in ('.yaml','.yml','.json')}
        self.metadata=dict(self.baseline.metadata,output='whole-text character spans',
            scope='local CredSweeper proxy; NOT deployed PromptGuard ensemble',
            match_policy='exact character spans; containment reported separately',
            config_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            gitleaks='NOT EVALUATED',custom_rules='NOT EVALUATED',mapping_policy='value offsets, no string search')

    def scan_text(self,text,source='configuration'):
        from credsweeper.file_handler.string_content_provider import StringContentProvider
        suffix={'code':'.py','configuration':'.env','markdown':'.md'}.get(source,'.txt')
        lines=text.split('\n');offsets=[];cursor=0
        for line in lines: offsets.append(cursor);cursor+=len(line)+1
        provider=StringContentProvider(lines,file_path='synthetic'+suffix,file_type=suffix)
        scanner=self.baseline.scanner;result=[]
        with private_scanner_logs():
            scanner.credential_manager.clear_credentials()
            try:
                scanner.scan([provider],None);scanner.post_processing(None)
                for candidate in scanner.credential_manager.get_credentials():
                    for ld in candidate.line_data_list:
                        n,a,b=ld.line_num,ld.value_start,ld.value_end
                        if not (type(n) is int and 1<=n<=len(lines) and 0<=a<b<=len(lines[n-1])):
                            raise ValueError('unmappable rule finding')
                        result.append(dict(start=offsets[n-1]+a,end=offsets[n-1]+b,detector='credsweeper',secret_type='SECRET'))
            except Exception:
                raise RuntimeError('rule scan or offset mapping failed; not a negative result') from None
            finally: scanner.credential_manager.clear_credentials()
        validate_spans(text,result)
        return list({(s['start'],s['end']):s for s in result}.values())
