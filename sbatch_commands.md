# sbatch commands


## Data reduction pipeline:

```
sbatch --job-name=webb-long-pipeline --output=webb-long-pipeline-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-LONG.py"
sbatch --job-name=webb-short-pipeline --output=web-short-pipeline-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-SHORT.py"
```

Modular pipeline runs:
```
sbatch --job-name=webb-long-pipeF405N --output=webb-long-pipeline-F405N-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-LONG.py --filternames=F405N"
sbatch --job-name=webb-long-pipeF410M --output=webb-long-pipeline-F410M-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-LONG.py --filternames=F410M"
sbatch --job-name=webb-long-pipeF466N --output=webb-long-pipeline-F466N-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-LONG.py --filternames=F466N"
sbatch --job-name=webb-short-pipeF212N --output=web-short-pipeline-F212N-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-SHORT.py --filternames=F212N"
sbatch --job-name=webb-short-pipeF187N --output=web-short-pipeline-F187N-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-SHORT.py --filternames=F187N"
sbatch --job-name=webb-short-pipeF182M --output=web-short-pipeline-F182M-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-SHORT.py --filternames=F182M"
```

Modular pipeline runs, only merged:
```
sbatch --job-name=webb-long-pipeF405Nmrg --output=webb-long-pipeline-F405N-merged-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-LONG.py --filternames=F405N --modules=merged"
sbatch --job-name=webb-long-pipeF410Mmrg --output=webb-long-pipeline-F410M-merged-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-LONG.py --filternames=F410M --modules=merged"
sbatch --job-name=webb-long-pipeF466Nmrg --output=webb-long-pipeline-F466N-merged-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-LONG.py --filternames=F466N --modules=merged"
sbatch --job-name=webb-short-pipeF212Nmrg --output=web-short-pipeline-F212N-merged-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-SHORT.py --filternames=F212N --modules=merged"
sbatch --job-name=webb-short-pipeF187Nmrg --output=web-short-pipeline-F187N-merged-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-SHORT.py --filternames=F187N --modules=merged"
sbatch --job-name=webb-short-pipeF182Mmrg --output=web-short-pipeline-F182M-merged-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/PipelineRerunNIRCAM-SHORT.py --filternames=F182M --modules=merged"
```

If saturated star finding needs to be run independently:
```
sbatch --job-name=brick-satstars --output=brick-satstars-%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/ipython /blue/adamginsburg/adamginsburg/jwst/brick/reduction/saturated_star_finding.py"
```

## Cataloging 

Individual filters:
```
sbatch --job-name=webb-cat-F182M --output=web-cat-F182M%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_short.py --filternames=F182M"
sbatch --job-name=webb-cat-F187N --output=web-cat-F187N%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_short.py --filternames=F187N"
sbatch --job-name=webb-cat-F212N --output=web-cat-F212N%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_short.py --filternames=F212N"
sbatch --job-name=webb-cat-F405N --output=web-cat-F405N%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F405N"
sbatch --job-name=webb-cat-F410M --output=web-cat-F410M%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F410M"
sbatch --job-name=webb-cat-F466N --output=web-cat-F466N%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F466N"
```

Bigger groups:
```
sbatch --job-name=webb-cat-long --output=web-cat-long%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py"
sbatch --job-name=webb-cat-short --output=web-cat-short%j.log --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_short.py"
sbatch --job-name=webb-cat-merge --output=web-cat-merge%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py"
```

With extra options:
```
sbatch --job-name=webb-cat-F405N-a --output=web-cat-F405N-a%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F405N --modules=nrca --desaturated=False"
sbatch --job-name=webb-cat-F410M-a --output=web-cat-F410M-a%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F410M --modules=nrca"
sbatch --job-name=webb-cat-F410M-b --output=web-cat-F410M-b%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F410M --modules=nrcb"
sbatch --job-name=webb-cat-F466N-a --output=web-cat-F466N-a%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F466N --modules=nrca --desaturated=False"
sbatch --job-name=webb-cat-F466N-a --output=web-cat-F466N-a%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=F466N --modules=nrca"
```