import os
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import hf_hub_download, snapshot_download
for repo in ['BeIR/nq','BeIR/hotpotqa','BeIR/msmarco']:
    for a in range(10):
        try:
            p=hf_hub_download(repo,'corpus/corpus-00000-of-00001.parquet',repo_type='dataset'); print(repo,'->',p,flush=True); break
        except Exception as e: print(repo,'retry',a,type(e).__name__,flush=True)
print('contriever ->', snapshot_download('facebook/contriever'),flush=True)
print('DONE',flush=True)
