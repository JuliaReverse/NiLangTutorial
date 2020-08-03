### A Pluto.jl notebook ###
# v0.11.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e54a1be6-d485-11ea-0262-034c56e0fda8
md"""
## Reversible functions

A reversible function `f` is defined as
```julia
(~f)(f(x, y, z...)...) == (x, y, z...)
```

e.g.
```math
\begin{align}
f &: x, y → x+y, y\\
{\small \mathrel{\sim}}f &: x, y → x-y, y
\end{align}
```
"""

# ╔═╡ 55a3a260-d48e-11ea-06e2-1b7bd7bba6f5
md"""
## Automatic differentiation in NiLang
![adprog](https://github.com/GiggleLiu/NiLang.jl/raw/master/docs/src/asset/adprog.png)
"""

# ╔═╡ 102fbf2e-d56b-11ea-189d-c78d56c0a924
html"""
<div>
<div style="float: left"><img src="https://adbenchresults.blob.core.windows.net/plots/2020-03-29_15-46-08_70e2e936bea81eebf0de78ce18d4d196daf1204e/static/jacobian/BA%20[Jacobian]%20-%20Release%20Graph.png" width=600/></div>
<div style=""><img src="https://adbenchresults.blob.core.windows.net/plots/2020-03-29_15-46-08_70e2e936bea81eebf0de78ce18d4d196daf1204e/static/jacobian/GMM%20(10k)%20[Jacobian]%20-%20Release%20Graph.png" width=600/></div>
</div>
"""

# ╔═╡ 2baaff10-d56c-11ea-2a23-bfa3a7ae2e4b
md"""
*Srajer, Filip, Zuzana Kukelova, and Andrew Fitzgibbon. "A benchmark of selected algorithmic differentiation tools on some problems in computer vision and machine learning." Optimization Methods and Software 33.4-6 (2018): 889-906.*

**Github Repos** 
* [https://github.com/microsoft/ADBench](https://github.com/microsoft/ADBench)
* [https://github.com/JuliaReverse/NiBundleAdjustment.jl](https://github.com/JuliaReverse/NiBundleAdjustment.jl)
* [https://github.com/JuliaReverse/NiGaussianMixture.jl](https://github.com/JuliaReverse/NiGaussianMixture.jl)
"""

# ╔═╡ dfa98e2a-d49d-11ea-27ce-a9542212bdbb
md"""
## Reversible Control Flow
"""

# ╔═╡ d4710366-d49e-11ea-0265-6929049649be
html"""
<h5>For loop</h5>
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2; -webkit-column-rule: 1px dotted #e0e0e0; -moz-column-rule: 1px dotted #e0e0e0; column-rule: 1px dotted #e0e0e0; margin-top:30px">
<div style="display: inline-float">
	<center><strong>Forward</strong></center>
	<pre><code class="language-julia">
	for i=start:step:stop
		# do something
	end
	</code></pre>
</div>
<div style="display: inline-block;">
	<center><strong>Reverse</strong></center>
	<pre><code class="language-julia">
	for i=stop:-step:start
		# undo something
	end
	</code>
	</pre>
</div>
</div>
"""

# ╔═╡ 0f2e256c-d4a2-11ea-0995-bbc54536f498
html"""
<h5>If statement</h5>
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2; -webkit-column-rule: 1px dotted #e0e0e0; -moz-column-rule: 1px dotted #e0e0e0; column-rule: 1px dotted #e0e0e0; margin-top:30px">
<div style="display: inline-float">
	<center><strong>Forward</strong></center>
	<pre><code class="language-julia">
	if (precondition, postcondition)
		# do A
	else
		# do B
	end
	</code></pre>
</div>
<div style="display: inline-block;">
	<center><strong>Reverse</strong></center>
	<pre><code class="language-julia">
	if (postcondition, precondition)
		# undo A
	else
		# undo B
	end
	</code>
	</pre>
</div>
</div>
"""

# ╔═╡ 62b9c7a6-d4a2-11ea-1939-dd76c69ae99c
html"""
<h5>While statement</h5>
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2; -webkit-column-rule: 1px dotted #e0e0e0; -moz-column-rule: 1px dotted #e0e0e0; column-rule: 1px dotted #e0e0e0; margin-top:30px">
<div style="display: inline-float">
	<center><strong>Forward</strong></center>
	<pre><code class="language-julia">
	while (precondition, postcondition)
		# do something
	end
	</code></pre>
</div>
<div style="display: inline-block;">
	<center><strong>Reverse</strong></center>
	<pre><code class="language-julia">
	while (postcondition, precondition)
		# undo something
	end
	</code>
	</pre>
</div>
</div>
"""

# ╔═╡ 11ddebfe-d488-11ea-223a-e9403f6ec8de
md"""
## Reversible BLAS functions

Affine transformation as an example

```julia
y = A * x + b
```
"""

# ╔═╡ 5f1c3f6c-d48b-11ea-3eb0-357fd3ece4fc
md"""
### Rounding error and reversible number systems

* Integers are reversible under (`+=`, `-=`).
* Floating point number system is **irreversible** under (`+=`, `-=`) and (`*=`, `/=`).
* [Fixedpoint number system](https://github.com/JuliaMath/FixedPointNumbers.jl) are reversible under (`+=`, `-=`)
* [Logarithmic number system](https://github.com/cjdoris/LogarithmicNumbers.jl) is reversible under (`*=`, `/=`)
"""

# ╔═╡ 259a2852-d48c-11ea-0f01-b9634850e09d
md"""
## Reversible arithmetic functions

Computing basic functions like `power`, `exp` and `besselj` is not trivial for reversible programming.
There is no efficient constant memory algorithm using pure fixed point numbers only.
For example, to compute `x ^ n` reversiblly with fixed point numbers,
we need to allocate a vector of size $O(n)$.
With logarithmic numbers, the above computation is straight forward.
"""

# ╔═╡ 02fb8e62-d4a3-11ea-2a6e-bd415591c891
md"""
## Applications
"""

# ╔═╡ 59797cc6-d4a3-11ea-03ab-610102dbc549
md"""
##### 1. Solve the memory wall problem in machine learning
"""

# ╔═╡ b44e12b8-d4a3-11ea-3f55-776476cd7d69
md"""
##### 3. Optimizing problems in finance
Gradient based optimization of Sharpe rate.

**Reference**
Han Li's Github repo: [https://github.com/HanLi123/NiLang](https://github.com/HanLi123/NiLang)
and his Zhihu blog [猴子掷骰子](https://zhuanlan.zhihu.com/c_1092471228488634368).
"""

# ╔═╡ 89de0b7e-d4a2-11ea-278a-a392b1649486
md"""
## Reversible hardwares
"""

# ╔═╡ 9e509f80-d485-11ea-0044-c5b7e750aacb
using NiLang, PlutoUI

# ╔═╡ 8064ce1c-d492-11ea-3e9a-b9284ee55ff3
using BenchmarkTools

# ╔═╡ a28d38be-d486-11ea-2c40-a377b74a05c1
# NiLang implementation
@i function f(x, y)
	x += y
end

# ╔═╡ e93f0bf6-d487-11ea-1baa-21d51ddb4a20
f(2, 3)

# ╔═╡ fc932606-d487-11ea-303e-75ca8b7a02f6
(~f)(5, 3)

# ╔═╡ 63b31d9c-d491-11ea-1e7a-cdf81d8e7ae7
function f2(x::AbstractArray{T}) where T
	res = zero(T)
	for i=1:length(x)
		@inbounds res += x[i]^2
	end
	sqrt(res)
end

# ╔═╡ 3b4a6b0a-d491-11ea-25db-ad48a3d17662
@i function rev_f2(res, y, x::AbstractArray{T}) where {T}
	for i=1:length(x)
		@inbounds y += x[i]^2
	end
	res += sqrt(y)
end

# ╔═╡ dfb0d542-d491-11ea-1791-b157ce622ba5
v = randn(10)

# ╔═╡ 37ed073a-d492-11ea-156f-1fb155128d0f
begin
	using Zygote
	Zygote.gradient(f2, v)
end

# ╔═╡ 4d75f302-d492-11ea-31b9-bbbdb43f344e
begin
	using NiLang.AD
	NiLang.AD.gradient(rev_f2, (0.0, 0.0, v), iloss=1)
end

# ╔═╡ da05bc8e-d491-11ea-0912-f9fc91583154
f2(v)

# ╔═╡ ed27aaf2-d491-11ea-369d-cd9f67bab59a
rev_f2(0.0, 0.0, v)

# ╔═╡ 744dd3c6-d492-11ea-0ed5-0fe02f99db1f
@benchmark Zygote.gradient($f2, $(randn(1000)))

# ╔═╡ 8ad60dc0-d492-11ea-2cb3-1750b39ddf86
@benchmark NiLang.AD.gradient($rev_f2, (0.0, 0.0, $(randn(1000))), iloss=1)

# ╔═╡ fcca27ba-d4a4-11ea-213a-c3e2305869f1
#**1. The bundle adjustment jacobian benchmark**
#$(LocalResource("ba-origin.png"))
#![ba](https://github.com/JuliaReverse/NiBundleAdjustment.jl/raw/master/benchmarks/adbench.png)

#**2. The Gaussian mixture model benchmark**
#$(LocalResource("gmm-origin.png"))
#![gmm](https://github.com/JuliaReverse/NiGaussianMixture.jl/raw/master/benchmarks/adbench.png)

md"""
## Performance

$(LocalResource("asset/benchmarks.png"))
"""

# ╔═╡ 2a17f2f6-d496-11ea-1c31-5f85dfd10488
md"""
## Memory Management

Reversible memory allocation is nontrivial because assign (`=`) is not allowed.

##### Memory Allocation
```julia
x ← constant
@zeros T x y z...
```

Basically equivalent to the assign (`=`) operation, but has reverse.

##### Memory Deallocation
```julia
x → constant
~@zeros T x y z...
```
(Note: symbols not in argument list or not deallocated manually will be deallocated automatically.)

##### Ancilla variables
Helper variables that allocated and deallocated inside a reversible function.
```julia
ancilla ← 0.0
# do something
ancilla → 0.0
```

##### Stack operations
```julia
PUSH!(stack, var)
PUSH!(var)  # push `var` to the global stack
```
`PUSH!` zero clears `var`.

```julia
POP!(stack, var)
POP!(var)  # pop `var` from the global stack
```
`POP!` preassumes `var` is zero cleared.

##### Compute-copy-uncompute paradigm
1. compute desired output,
2. copy the result to an emptied memory,
3. undo the computation to restore the values of variables, especially ancillas.

$(LocalResource("asset/compute-copy-uncompute.png", :width=>500))
"""

# ╔═╡ 8a8aaea2-d49c-11ea-2014-37718ebe6465
@i function rev_f2_v2(res, x::AbstractArray{T}) where {T}
	y ← zero(T)
	@routine for i=1:length(x)
		@inbounds y += x[i]^2
	end
	res += sqrt(y)
	~@routine
end

# ╔═╡ 90355e92-d49c-11ea-3d8e-a78925bb4c41
rev_f2_v2(0.0, v)

# ╔═╡ fe333e28-d49c-11ea-3c5f-3f9fdccfb00c
@i function rev_f2_v3(res, x::AbstractArray{T}) where {T}
	y ← zero(T)
	@routine for i=1:length(x)
		@inbounds y += x[i]^2
	end
	res += sqrt(y)
	PUSH!(y)
end

# ╔═╡ 0d409398-d49d-11ea-2927-555c9923dfbd
rev_f2_v3(0.0, v)

# ╔═╡ 19132a62-d49d-11ea-277b-0782a361aa4b
NiLang.AD.gradient(rev_f2_v3, (0.0, v), iloss=1)

# ╔═╡ a0e231f0-d4b2-11ea-3eac-e34f4afbabe6
LocalResource("asset/if.png", :width=>400)

# ╔═╡ 58495bbc-d4b2-11ea-0e7a-2f5f4a0596a9
LocalResource("asset/while.png", :width=>400)

# ╔═╡ 030e592e-d488-11ea-060d-97a3bb6353b7
@i function reversible_affine!(y!::AbstractVector{T}, W::AbstractMatrix{T}, b::AbstractVector{T}, x::AbstractVector{T}) where T
    @safe @assert size(W) == (length(y!), length(x)) && length(b) == length(y!)
    for j=1:size(W, 2)
        for i=1:size(W, 1)
            @inbounds y![i] += W[i,j]*x[j]
        end
    end
    for i=1:size(W, 1)
        @inbounds y![i] += b[i]
    end
end

# ╔═╡ c8d26856-d48a-11ea-3cd3-1124cd172f3a
begin
	W = randn(10, 10)
	b = randn(10)
	x = randn(10)
end;

# ╔═╡ 37c4394e-d489-11ea-174c-b13bdddbe741
yout, Wout, bout, xout = reversible_affine!(zeros(10), W, b, x)

# ╔═╡ fef54688-d48a-11ea-340b-295b88d21382
yin, Win, bin, xin = (~reversible_affine!)(yout, Wout, bout, xout)

# ╔═╡ 5a8ba8f4-d493-11ea-1839-8ba81f86799d
@i function i_power(y::T, x::T, n::Int) where T
    @routine begin
        lx ← one(ULogarithmic{T})
        ly ← one(ULogarithmic{T})
        ## convert `x` to a logarithmic number
        ## Here, `*=` is reversible for log numbers
        lx *= convert(x)
        for i=1:n
            ly *= lx
        end
    end

    ## convert back to fixed point numbers
    y += convert(ly)

    ~@routine
end

# ╔═╡ a625a922-d493-11ea-1fe9-bdd4a694cde0
# To check the function
i_power(Fixed43(0.0), Fixed43(0.4), 3)

# ╔═╡ 85edd210-d493-11ea-018f-3599efa549b0
@i function i_exp(y!::T, x::T) where T<:Union{Fixed43, GVar{<:Fixed43}}
	s ← one(ULogarithmic{T})
	lx ← one(ULogarithmic{T})
	k ← 0
	lx *= convert(x)
	y! += convert(s)
	while (s.log > -20, k != 0)
		k += 1
		s *= lx / k
		y! += convert(s)
	end
	~(while (s.log > -20, k != 0)
		k += 1
		s *= x / k
	end)
	lx /= convert(x)
end

# ╔═╡ 43349304-d494-11ea-1691-29b02c7867b4
let x = Fixed43(3.5)
	# We can check the reversibility
	out, _ = i_exp(Fixed43(0.0), x)
	@assert out ≈ exp(3.5)

	# Computing the gradients
	_, gx = NiLang.AD.gradient(Val(1), i_exp, (Fixed43(0.0), x))
	(out, gx)
end

# ╔═╡ 7aa1d85e-d4a7-11ea-1807-6fcb8fdf2973
begin
	x1 = [Fixed43(3.5)]
	y1 = [Fixed43(0.0)]
	s1 = [one(ULogarithmic{Fixed43})]
	lx1 = [one(ULogarithmic{Fixed43})]
	k1 = [0]
	lx1[] *= convert(ULogarithmic{Fixed43}, x1[])
	y1[] += convert(Fixed43, s1[])
end

# ╔═╡ f1fb7290-d4a8-11ea-381d-43a97aa5f408
@i function exp_step(y, lx, k, s)
	k[] += 1
	s[] *= lx[] / k[]
	y[] += convert(s[])
end

# ╔═╡ 7abf09e6-d495-11ea-09c1-d36b9fbc29ef
md"""
$(@bind left html"<button><</button>")
$(@bind right html"<button>></button>")
"""

# ╔═╡ 91b1bf06-d495-11ea-2930-9b3a0b075c9e
begin
	right
	exp_step(y1, lx1, k1, s1)
	nothing
end

# ╔═╡ 925ad4fc-d4a8-11ea-17eb-b72ab2807ae0
let
	left
	(~exp_step)(y1, lx1, k1, s1)
	nothing
end

# ╔═╡ 95677526-d4b0-11ea-04c8-c35d4f62c3a0
begin
	left
	right
	# should be table https://github.com/JuliaLang/julia/issues/16194
	md"""y1 = $(y1[])   ,  lx1 = $(lx1[])
	
	k1 = $(k1[]), s1 = $(s1[])
	"""
end

# ╔═╡ 737b7440-d4a3-11ea-35ee-27a2b1b2ee35
md"""
##### 2. Solve hard scientific problems
Obtaining the optimal configuration of a spinglass problem on a $28 \times 28$ square lattice.

$(LocalResource("asset/spinglass28.svg", :width=>400))
"""

# ╔═╡ Cell order:
# ╟─e54a1be6-d485-11ea-0262-034c56e0fda8
# ╠═9e509f80-d485-11ea-0044-c5b7e750aacb
# ╠═a28d38be-d486-11ea-2c40-a377b74a05c1
# ╠═e93f0bf6-d487-11ea-1baa-21d51ddb4a20
# ╠═fc932606-d487-11ea-303e-75ca8b7a02f6
# ╟─55a3a260-d48e-11ea-06e2-1b7bd7bba6f5
# ╠═63b31d9c-d491-11ea-1e7a-cdf81d8e7ae7
# ╠═3b4a6b0a-d491-11ea-25db-ad48a3d17662
# ╠═dfb0d542-d491-11ea-1791-b157ce622ba5
# ╠═da05bc8e-d491-11ea-0912-f9fc91583154
# ╠═ed27aaf2-d491-11ea-369d-cd9f67bab59a
# ╠═37ed073a-d492-11ea-156f-1fb155128d0f
# ╠═4d75f302-d492-11ea-31b9-bbbdb43f344e
# ╠═8064ce1c-d492-11ea-3e9a-b9284ee55ff3
# ╠═744dd3c6-d492-11ea-0ed5-0fe02f99db1f
# ╠═8ad60dc0-d492-11ea-2cb3-1750b39ddf86
# ╟─fcca27ba-d4a4-11ea-213a-c3e2305869f1
# ╟─102fbf2e-d56b-11ea-189d-c78d56c0a924
# ╟─2baaff10-d56c-11ea-2a23-bfa3a7ae2e4b
# ╟─2a17f2f6-d496-11ea-1c31-5f85dfd10488
# ╠═8a8aaea2-d49c-11ea-2014-37718ebe6465
# ╠═90355e92-d49c-11ea-3d8e-a78925bb4c41
# ╠═fe333e28-d49c-11ea-3c5f-3f9fdccfb00c
# ╠═0d409398-d49d-11ea-2927-555c9923dfbd
# ╠═19132a62-d49d-11ea-277b-0782a361aa4b
# ╟─dfa98e2a-d49d-11ea-27ce-a9542212bdbb
# ╟─d4710366-d49e-11ea-0265-6929049649be
# ╟─0f2e256c-d4a2-11ea-0995-bbc54536f498
# ╟─a0e231f0-d4b2-11ea-3eac-e34f4afbabe6
# ╟─62b9c7a6-d4a2-11ea-1939-dd76c69ae99c
# ╟─58495bbc-d4b2-11ea-0e7a-2f5f4a0596a9
# ╟─11ddebfe-d488-11ea-223a-e9403f6ec8de
# ╠═030e592e-d488-11ea-060d-97a3bb6353b7
# ╠═c8d26856-d48a-11ea-3cd3-1124cd172f3a
# ╠═37c4394e-d489-11ea-174c-b13bdddbe741
# ╠═fef54688-d48a-11ea-340b-295b88d21382
# ╟─5f1c3f6c-d48b-11ea-3eb0-357fd3ece4fc
# ╟─259a2852-d48c-11ea-0f01-b9634850e09d
# ╠═5a8ba8f4-d493-11ea-1839-8ba81f86799d
# ╠═a625a922-d493-11ea-1fe9-bdd4a694cde0
# ╠═85edd210-d493-11ea-018f-3599efa549b0
# ╠═43349304-d494-11ea-1691-29b02c7867b4
# ╠═7aa1d85e-d4a7-11ea-1807-6fcb8fdf2973
# ╠═f1fb7290-d4a8-11ea-381d-43a97aa5f408
# ╟─7abf09e6-d495-11ea-09c1-d36b9fbc29ef
# ╠═91b1bf06-d495-11ea-2930-9b3a0b075c9e
# ╠═925ad4fc-d4a8-11ea-17eb-b72ab2807ae0
# ╠═95677526-d4b0-11ea-04c8-c35d4f62c3a0
# ╟─02fb8e62-d4a3-11ea-2a6e-bd415591c891
# ╠═59797cc6-d4a3-11ea-03ab-610102dbc549
# ╟─737b7440-d4a3-11ea-35ee-27a2b1b2ee35
# ╠═b44e12b8-d4a3-11ea-3f55-776476cd7d69
# ╠═89de0b7e-d4a2-11ea-278a-a392b1649486
