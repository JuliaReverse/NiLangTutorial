# NiLang Tutorial

NOTE: migrated to [here](https://giggleliu.github.io/NiLang.jl/dev/notebooks/basic.html), this repo is not maintained anymore.

This repository contains Pluto notebooks for learning [NiLang](https://github.com/GiggleLiu/NiLang.jl).

## Using Notebooks
Open the notebooks ended with `.jl` in `notebooks/` folder with [Pluto](https://github.com/fonsp/Pluto.jl).

Or simply paste the following code into your terminal
```bash
$ git clone https://github.com/JuliaReverse/NiLangTutorial.git
$ julia --project='NiLangTutorial/notebooks/Project.toml' -e \
'using Pluto; Pluto.run(notebook="NiLangTutorial/notebooks/basic.jl")'
```
(NOTE: you need to set julia path correctly first: https://julialang.org/downloads/platform/)

At the time of writting this tutorial, the NiLang version is 0.7.1.

