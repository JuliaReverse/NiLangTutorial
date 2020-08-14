### A Pluto.jl notebook ###
# v0.11.6

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

# ╔═╡ 55cfdab8-d792-11ea-271f-e7383e19997c
using PlutoUI;

# ╔═╡ 9e509f80-d485-11ea-0044-c5b7e750aacb
using NiLang, PrettyTables

# ╔═╡ 8064ce1c-d492-11ea-3e9a-b9284ee55ff3
using BenchmarkTools

# ╔═╡ a11c4b60-d77d-11ea-1afe-1f2ab9621f42
md"""
# 连猩猩都能懂的可逆编程: A Niang Tutorial
**Jinguo Liu** (github: [GiggleLiu](https://github.com/GiggleLiu/))

*Postdoc, Institute of physics, Chinese academy of sciences* (when doing this project)

*Consultant, QuEra Computing* (current)

*Postdoc, Havard* (soon)

## Table of Contents
1. Reversible programming basic: Reversible functions
2. Automatic differentiation in NiLang
3. Reversible memory management and control flow
4. A benchmark with Tapenade et al.
5. Reversible programming without rounding error
6. Real world applications
7. The hardware and the future

In this talk,
We use the reversible eDSL ![NiLang](https://raw.githubusercontent.com/GiggleLiu/NiLang.jl/master/docs/src/asset/logo3.png) as our reversible programming tool.

[NiLang](https://github.com/GiggleLiu/NiLang.jl) is a [Julia](https://julialang.org/) package that features
* a reversible language with arrays,
* open source,
* speed,
* reversible logarithmic number system,
* complex valued AD
"""

# ╔═╡ e54a1be6-d485-11ea-0262-034c56e0fda8
md"""
## Sec I. Reversible programming basic: Reversible functions

A reversible function `f` is defined as
```julia
(~f)(f(x, y, z...)...) == (x, y, z...)
```

####  Example 1: reversible adder
```math
\begin{align}
f &: x, y → x+y, y\\
{\small \mathrel{\sim}}f &: x, y → x-y, y
\end{align}
```
"""

# ╔═╡ a28d38be-d486-11ea-2c40-a377b74a05c1
# NiLang implementation
@i function reversible_plus(x, y)
	x += y
end

# ╔═╡ e93f0bf6-d487-11ea-1baa-21d51ddb4a20
reversible_plus(2.0, 3.0)

# ╔═╡ fc932606-d487-11ea-303e-75ca8b7a02f6
(~reversible_plus)(5.0, 3.0)

# ╔═╡ 05e91f18-ddf1-11ea-105b-530556566fd7
md"**Comment 1**: The reversible macro `@i` defines two functions, the function itself and its inverse."

# ╔═╡ a961e048-ddf2-11ea-0262-6d19eb82b36b
md"**Comment 2**: The return statement is not allowed, a reversible function returns input arguments directly."

# ╔═╡ 2d22f504-ddf1-11ea-28ec-5de6f4ee79bb
md"**Comment 3**: `+=` is considered as reversible for integers and floating point numbers in NiLang, although for floating point numbers, there are *rounding errors*."

# ╔═╡ 0b4edb1a-ddf0-11ea-220c-91f2df7452e7
@i function reversible_plus2(x, y)
	reversible_plus(x, y)  # equivalent to `reversible_plus(x, y)`
	reversible_plus(x, y)
end

# ╔═╡ f875ecd6-ddef-11ea-22a1-619809d15b37
md"**Comment 4**: Inside a reversible function definition, a statement changes a variable *inplace*"

# ╔═╡ 913af55a-ddef-11ea-3715-259cf0454ce6
md"One can execute a single statement in a reversible function using `@instr`"

# ╔═╡ 9028f6b4-ddef-11ea-3130-cf182138d0b8
let
	x, y = 2, 3
	@instr reversible_plus(x, y)
	(x, y)
end

# ╔═╡ 712c0fa6-d78e-11ea-2bcb-f3e60bf3c55d
md"#### Example 2: reversible norm function"

# ╔═╡ 39720bc4-d77a-11ea-323a-a72ddf18d94a
function regular_norm(x::AbstractArray{T}) where T
	res = zero(T)  # !
	for i=1:length(x)
		@inbounds res += x[i]^2
	end
	return sqrt(res) # !
end

# ╔═╡ 3b4a6b0a-d491-11ea-25db-ad48a3d17662
@i function reversible_norm(res, y, x::AbstractArray{T}) where {T}
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
	Zygote.gradient(regular_norm, v)
end

# ╔═╡ 4d75f302-d492-11ea-31b9-bbbdb43f344e
begin
	using NiLang.AD
	NiLang.AD.gradient(reversible_norm, (0.0, 0.0, v), iloss=1)
end

# ╔═╡ da05bc8e-d491-11ea-0912-f9fc91583154
regular_norm(v)

# ╔═╡ ed27aaf2-d491-11ea-369d-cd9f67bab59a
reversible_norm(0.0, 0.0, v)

# ╔═╡ 7a6b9cb4-d77d-11ea-23e4-81537a71a23f
md"""
**Reversibility explained**
"""

# ╔═╡ 3bbf443a-d77c-11ea-1cdd-81b5a00bbd85
ex = :(begin
	for i=1:length(x)
		@inbounds y += x[i]^2
	end
	res += sqrt(y)
	end); NiLangCore.rmlines(ex)

# ╔═╡ 83054e30-d79c-11ea-39a3-ebf3c538e39c
md"**Obtain the reversed statement**"

# ╔═╡ db16bc76-d77b-11ea-33b2-c10277f30055
reversed_ex = NiLangCore.dual_ex(Main, ex); reversed_ex |> NiLangCore.rmlines

# ╔═╡ b8dcfb90-d79c-11ea-316f-439123219eee
md"**Compile the reversible IR to native julia code**"

# ╔═╡ b1528b26-d77c-11ea-2368-8b95a5a39137
NiLangCore.compile_ex(Main, ex) |> NiLangCore.rmlines

# ╔═╡ 67c0adac-d77d-11ea-2093-8d4b3c00d0fe
NiLangCore.compile_ex(Main, reversed_ex) |> NiLangCore.rmlines

# ╔═╡ 55d246dc-ddef-11ea-3eb9-e3a3aecf8ac8
md"##### Example 3: Complex valued log"

# ╔═╡ 7a14db2c-ddef-11ea-02d5-413141f8d614
@i @inline function (:+=)(log)(y!::Complex{T}, x::Complex{T}) where T
	@routine begin
		n ← zero(T)
		n += abs(x)
	end
	y!.re += log(n)
	y!.im += angle(x)
	~@routine
end

# ╔═╡ 8345cf8a-ddef-11ea-0af6-dfbdc737adbc


# ╔═╡ d5c2efbc-d779-11ea-11ad-1f5873b95628
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ 55a3a260-d48e-11ea-06e2-1b7bd7bba6f5
md"""
## Sec II. Automatic differentiation in NiLang
![adprog](https://github.com/GiggleLiu/NiLang.jl/raw/master/docs/src/asset/adprog.png)
"""

# ╔═╡ 2e6fe4da-d79d-11ea-1e90-f5215190395c
md"**Obtaining the gradient of the norm function**"

# ╔═╡ 744dd3c6-d492-11ea-0ed5-0fe02f99db1f
@benchmark Zygote.gradient($regular_norm, $(randn(1000))) seconds=1

# ╔═╡ 8ad60dc0-d492-11ea-2cb3-1750b39ddf86
@benchmark NiLang.AD.gradient($reversible_norm, (0.0, 0.0, $(randn(1000))), iloss=1)

# ╔═╡ 7753c622-d77e-11ea-1442-037197af6296
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ 2a17f2f6-d496-11ea-1c31-5f85dfd10488
md"""
## Sec III. Reversible memory management and control flow

### 1. Memory Management

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
"""

# ╔═╡ 023193dc-d78e-11ea-2e17-e54ad9144b91
md"""
#### Example 1: norm with stack operations
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
"""

# ╔═╡ d0555864-d78d-11ea-0704-73715bbd9c08
md"""
#### Example 2: norm with compute-copy-uncompute paradigm
1. compute desired output,
2. copy the result to an emptied memory,
3. undo the computation to restore the values of variables, especially ancillas.

$(LocalResource("asset/compute-copy-uncompute.png", :width=>500))
"""

# ╔═╡ dfa98e2a-d49d-11ea-27ce-a9542212bdbb
md"""
### 2. Reversible Control Flow
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

# ╔═╡ a0e231f0-d4b2-11ea-3eac-e34f4afbabe6
LocalResource("asset/if.png", :width=>400)

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

# ╔═╡ 58495bbc-d4b2-11ea-0e7a-2f5f4a0596a9
LocalResource("asset/while.png", :width=>400)

# ╔═╡ 7bab4614-d77e-11ea-037c-8d1f432fc3b8
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ fcca27ba-d4a4-11ea-213a-c3e2305869f1
#**1. The bundle adjustment jacobian benchmark**
#$(LocalResource("ba-origin.png"))
#![ba](https://github.com/JuliaReverse/NiBundleAdjustment.jl/raw/master/benchmarks/adbench.png)

#**2. The Gaussian mixture model benchmark**
#$(LocalResource("gmm-origin.png"))
#![gmm](https://github.com/JuliaReverse/NiGaussianMixture.jl/raw/master/benchmarks/adbench.png)

md"""
## Sec IV. A benchmark with Tapenade et al.

Functions benchmarked
* Bundle Adjustment (Jacobian)
![bundle adjustment](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRgpGSCWRjHDDaIYQX5ejhMvyKY_GFhynVoQg&usqp=CAU)
* Gaussian Mixture Model (Gradient)
![gmm](https://prateekvjoshi.files.wordpress.com/2013/06/multimodal.jpg)
"""

# ╔═╡ 2baaff10-d56c-11ea-2a23-bfa3a7ae2e4b
md"""
*Srajer, Filip, Zuzana Kukelova, and Andrew Fitzgibbon. "A benchmark of selected algorithmic differentiation tools on some problems in computer vision and machine learning." Optimization Methods and Software 33.4-6 (2018): 889-906.*

**Devices**
* CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
* GPU: Nvidia Titan V. 

**Github Repos** 
* [https://github.com/microsoft/ADBench](https://github.com/microsoft/ADBench)
* [https://github.com/JuliaReverse/NiBundleAdjustment.jl](https://github.com/JuliaReverse/NiBundleAdjustment.jl)
* [https://github.com/JuliaReverse/NiGaussianMixture.jl](https://github.com/JuliaReverse/NiGaussianMixture.jl)
"""

# ╔═╡ 102fbf2e-d56b-11ea-189d-c78d56c0a924
html"""
<h5>Results from the original benchmark<h5>
<div>
<div style="float: left"><img src="https://adbenchresults.blob.core.windows.net/plots/2020-03-29_15-46-08_70e2e936bea81eebf0de78ce18d4d196daf1204e/static/jacobian/BA%20[Jacobian]%20-%20Release%20Graph.png" width=340/></div>
<div style=""><img src="https://adbenchresults.blob.core.windows.net/plots/2020-03-29_15-46-08_70e2e936bea81eebf0de78ce18d4d196daf1204e/static/jacobian/GMM%20(10k)%20[Jacobian]%20-%20Release%20Graph.png" width=340/></div>
</div>
"""

# ╔═╡ cc0d5622-d788-11ea-19cd-3bf6864d9263
md"""##### Including NiLang.AD
$(LocalResource("asset/benchmarks.png"))"""

# ╔═╡ 7c79975c-d789-11ea-30b1-67ff05418cdb
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ 5f1c3f6c-d48b-11ea-3eb0-357fd3ece4fc
md"""
## Sec V. Reversible programming without rounding error

 

**Appology**: I lied, `+=` is not reversible for floating point numbers

 

* Integers are reversible under (`+=`, `-=`).
* Floating point number system is **irreversible** under (`+=`, `-=`) and (`*=`, `/=`).
* [Fixedpoint number system](https://github.com/JuliaMath/FixedPointNumbers.jl) are reversible under (`+=`, `-=`)
* [Logarithmic number system](https://github.com/cjdoris/LogarithmicNumbers.jl) is reversible under (`*=`, `/=`)
"""

# ╔═╡ 11ddebfe-d488-11ea-223a-e9403f6ec8de
md"""
##### Example 1: Affine transformation with rounding error

```julia
y = A * x + b
```
"""

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
# should be restored to 0, but not!
yin, Win, bin, xin = (~reversible_affine!)(yout, Wout, bout, xout)

# ╔═╡ 259a2852-d48c-11ea-0f01-b9634850e09d
md"""
### Reversible arithmetic functions

Computing basic functions like `power`, `exp` and `besselj` is not trivial for reversible programming.
There is no efficient constant memory algorithm using pure fixed point numbers only.
"""

# ╔═╡ f06fb004-d79f-11ea-0d60-8151019bf8c7
md"""
##### Example 2: Computing power function
To compute `x ^ n` reversiblly with fixed point numbers,
we need to either allocate a vector of size $O(n)$ or suffer from polynomial time overhead. It does not show the advantage to checkpointing.
"""

# ╔═╡ 26a8a42c-d7a1-11ea-24a3-45bc6e0674ea
@i function i_power_cache(y!::T, x::T, n::Int) where T
    @routine @invcheckoff begin
        cache ← zeros(T, n)  # allocate a buffer of size n
		cache[1] += x
        for i=2:n
            cache[i] += cache[i-1] * x
        end
    end

    y! += cache[n]

    ~@routine  # uncompute cache
end

# ╔═╡ 399552c4-d7a1-11ea-36bb-ad5ca42043cb
# To check the function
i_power_cache(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 4bb19760-d7bf-11ea-12ed-4d9e4efb3482
md"""
##### Example 3: reversible thinker, the logarithmic number approach

With **logarithmic numbers**, we can still utilize reversibility. Fixed point numbers and logarithmic numbers can be converted via "a fast binary logarithm algorithm".

##### References
* [1] C. S. Turner, "A Fast Binary Logarithm Algorithm", IEEE Signal Processing Mag., pp. 124,140, Sep. 2010.
"""

# ╔═╡ 5a8ba8f4-d493-11ea-1839-8ba81f86799d
@i function i_power_lognumber(y::T, x::T, n::Int) where T
    @routine @invcheckoff begin
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
i_power_lognumber(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 4fd20ed2-d7a2-11ea-206e-13799234913f
md"**Less allocation, better speed**"

# ╔═╡ 692dfb44-d7a1-11ea-00da-af6550bc0622
@benchmark i_power_cache(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 7e4ee09c-d7a1-11ea-0e56-c1921012bc30
@benchmark i_power_lognumber(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 4c209bbe-d7b1-11ea-0628-33eb8d664f5b
md"""##### Example 4: The first kind Bessel function computed with Taylor expansion
```math
J_\nu(z) = \sum\limits_{n=0}^{\infty} \frac{(z/2)^\nu}{\Gamma(k+1)\Gamma(k+\nu+1)} (-z^2/4)^{n}
```


"""

# ╔═╡ fd44a3d4-d7a4-11ea-24ea-09456ff2c53d
@i function ibesselj(y!::T, ν, z::T; atol=1e-8) where T
	if z == 0
		if v == 0
			out! += 1
		end
	else
		@routine @invcheckoff begin
			k ← 0
			@ones ULogarithmic{T} lz halfz halfz_power_2 s
			@zeros T out_anc
			lz *= convert(z)
			halfz *= lz / 2
			halfz_power_2 *= halfz ^ 2
			# s *= (z/2)^ν/ factorial(ν)
			s *= halfz ^ ν
			for i=1:ν
				s /= i
			end
			out_anc += convert(s)
			while (s.log > -25, k!=0) # upto precision e^-25
				k += 1
				# s *= 1 / k / (k+ν) * (z/2)^2
				s *= halfz_power_2 / (k*(k+ν))
				if k%2 == 0
					out_anc += convert(s)
				else
					out_anc -= convert(s)
				end
			end
		end
		y! += out_anc
		~@routine
	end
end

# ╔═╡ 84272664-d7b7-11ea-2e37-dffd2023d8d6
md"z = $(@bind z html\"<input type=range value=0 min=0 max=10 step=0.1></input>\")"

# ╔═╡ 900e2ea4-d7b8-11ea-3511-6f12d95e638a
begin
	y = ibesselj(Fixed43(0.0), 2, Fixed43(z))[1]
	gz = NiLang.AD.gradient(Val(1), ibesselj, (Fixed43(0.0), 2, Fixed43(z)))[3]
end;

# ╔═╡ fe333e28-d49c-11ea-3c5f-3f9fdccfb00c
@i function reversible_norm_stack(res, x::AbstractArray{T}) where {T}
	y ← zero(T)  # allocate one element
	@routine for i=1:length(x)
		@inbounds y += x[i]^2
	end
	res += sqrt(y)
	PUSH!(y)   # store it into a stack
end

# ╔═╡ 19132a62-d49d-11ea-277b-0782a361aa4b
NiLang.AD.gradient(reversible_norm_stack, (0.0, v), iloss=1)

# ╔═╡ 0d409398-d49d-11ea-2927-555c9923dfbd
reversible_norm_stack(0.0, v)

# ╔═╡ 8a8aaea2-d49c-11ea-2014-37718ebe6465
@i function reversible_norm_uncompute(res, x::AbstractArray{T}) where {T}
	y ← zero(T)
	@routine for i=1:length(x)  # compute y
		@inbounds y += x[i]^2
	end
	res += sqrt(y)
	~@routine   # uncompute y, i.e. restore it to 0.
end

# ╔═╡ 5249aaa8-d78e-11ea-3853-d3b48a930ef8
NiLang.AD.gradient(reversible_norm_uncompute, (0.0, v), iloss=1)

# ╔═╡ 90355e92-d49c-11ea-3d8e-a78925bb4c41
reversible_norm_uncompute(0.0, v)

# ╔═╡ d76be888-d7b4-11ea-2989-2174682ead76
let
	str = pretty_table(String, round.(Float64[z y gz], digits=5), backend = :html, ["&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;z", "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y", "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;∂y/∂z"])
	HTML(str)
end

# ╔═╡ 8160f4a2-d789-11ea-28f8-e91d58a61642
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ 02fb8e62-d4a3-11ea-2a6e-bd415591c891
md"""
## Sec VI: Applications
"""

# ╔═╡ 6097b916-d92c-11ea-0dee-e9791b041b67
md"### 1. Solve the memory wall problem in machine learning"

# ╔═╡ 4a3f8c7c-d7bd-11ea-2370-3d6b629bc653
html"""
Learning a ring distribution with NICE network, before and after training

<img style="float:left" src="https://giggleliu.github.io/NiLang.jl/dev/asset/nice_before.png" width=340/>
<img src="https://giggleliu.github.io/NiLang.jl/dev/asset/nice_after.png" width=340/>

<h5>References</h5>
<ul>
<li><a href="https://arxiv.org/abs/1410.8516">arXiv: 1410.8516</li>
<li><a href="https://giggleliu.github.io/NiLang.jl/dev/examples/nice/#NICE-network-1">NiLang's documentation</a></li>
</ul>
"""

# ╔═╡ 737b7440-d4a3-11ea-35ee-27a2b1b2ee35
md"""
### 2. Solve hard scientific problems
Obtaining the optimal configuration of a spinglass problem on a $28 \times 28$ square lattice.

$(LocalResource("asset/spinglass28.svg", :width=>400))

##### References
unpublished
"""

# ╔═╡ b44e12b8-d4a3-11ea-3f55-776476cd7d69
md"""
### 3. Optimizing problems in finance
Gradient based optimization of Sharpe rate.

##### References
* Han Li's Github repo: [https://github.com/HanLi123/NiLang](https://github.com/HanLi123/NiLang) and his Zhihu blog [猴子掷骰子](https://zhuanlan.zhihu.com/c_1092471228488634368).
"""

# ╔═╡ 85c9edcc-d789-11ea-14c8-71697cd6a047
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ 89de0b7e-d4a2-11ea-278a-a392b1649486
md"""
## Sec VII: Reversible hardwares
"""

# ╔═╡ bc98ba6e-d7cc-11ea-0f63-79a720e8aa6c
md"""##### Adiabatic CMOS"""

# ╔═╡ 8a4d7fba-d789-11ea-2b3b-23f2c4e5cbdf
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ Cell order:
# ╟─a11c4b60-d77d-11ea-1afe-1f2ab9621f42
# ╟─e54a1be6-d485-11ea-0262-034c56e0fda8
# ╟─55cfdab8-d792-11ea-271f-e7383e19997c
# ╠═9e509f80-d485-11ea-0044-c5b7e750aacb
# ╠═a28d38be-d486-11ea-2c40-a377b74a05c1
# ╠═e93f0bf6-d487-11ea-1baa-21d51ddb4a20
# ╠═fc932606-d487-11ea-303e-75ca8b7a02f6
# ╟─05e91f18-ddf1-11ea-105b-530556566fd7
# ╟─a961e048-ddf2-11ea-0262-6d19eb82b36b
# ╟─2d22f504-ddf1-11ea-28ec-5de6f4ee79bb
# ╠═0b4edb1a-ddf0-11ea-220c-91f2df7452e7
# ╟─f875ecd6-ddef-11ea-22a1-619809d15b37
# ╟─913af55a-ddef-11ea-3715-259cf0454ce6
# ╠═9028f6b4-ddef-11ea-3130-cf182138d0b8
# ╟─712c0fa6-d78e-11ea-2bcb-f3e60bf3c55d
# ╠═39720bc4-d77a-11ea-323a-a72ddf18d94a
# ╠═3b4a6b0a-d491-11ea-25db-ad48a3d17662
# ╠═dfb0d542-d491-11ea-1791-b157ce622ba5
# ╠═da05bc8e-d491-11ea-0912-f9fc91583154
# ╠═ed27aaf2-d491-11ea-369d-cd9f67bab59a
# ╟─7a6b9cb4-d77d-11ea-23e4-81537a71a23f
# ╠═3bbf443a-d77c-11ea-1cdd-81b5a00bbd85
# ╟─83054e30-d79c-11ea-39a3-ebf3c538e39c
# ╠═db16bc76-d77b-11ea-33b2-c10277f30055
# ╟─b8dcfb90-d79c-11ea-316f-439123219eee
# ╠═b1528b26-d77c-11ea-2368-8b95a5a39137
# ╠═67c0adac-d77d-11ea-2093-8d4b3c00d0fe
# ╟─55d246dc-ddef-11ea-3eb9-e3a3aecf8ac8
# ╠═7a14db2c-ddef-11ea-02d5-413141f8d614
# ╠═8345cf8a-ddef-11ea-0af6-dfbdc737adbc
# ╟─d5c2efbc-d779-11ea-11ad-1f5873b95628
# ╟─55a3a260-d48e-11ea-06e2-1b7bd7bba6f5
# ╟─2e6fe4da-d79d-11ea-1e90-f5215190395c
# ╠═37ed073a-d492-11ea-156f-1fb155128d0f
# ╠═4d75f302-d492-11ea-31b9-bbbdb43f344e
# ╠═8064ce1c-d492-11ea-3e9a-b9284ee55ff3
# ╠═744dd3c6-d492-11ea-0ed5-0fe02f99db1f
# ╠═8ad60dc0-d492-11ea-2cb3-1750b39ddf86
# ╟─7753c622-d77e-11ea-1442-037197af6296
# ╟─2a17f2f6-d496-11ea-1c31-5f85dfd10488
# ╟─023193dc-d78e-11ea-2e17-e54ad9144b91
# ╠═fe333e28-d49c-11ea-3c5f-3f9fdccfb00c
# ╠═0d409398-d49d-11ea-2927-555c9923dfbd
# ╠═19132a62-d49d-11ea-277b-0782a361aa4b
# ╟─d0555864-d78d-11ea-0704-73715bbd9c08
# ╠═8a8aaea2-d49c-11ea-2014-37718ebe6465
# ╠═90355e92-d49c-11ea-3d8e-a78925bb4c41
# ╠═5249aaa8-d78e-11ea-3853-d3b48a930ef8
# ╟─dfa98e2a-d49d-11ea-27ce-a9542212bdbb
# ╟─d4710366-d49e-11ea-0265-6929049649be
# ╟─0f2e256c-d4a2-11ea-0995-bbc54536f498
# ╟─a0e231f0-d4b2-11ea-3eac-e34f4afbabe6
# ╟─62b9c7a6-d4a2-11ea-1939-dd76c69ae99c
# ╟─58495bbc-d4b2-11ea-0e7a-2f5f4a0596a9
# ╟─7bab4614-d77e-11ea-037c-8d1f432fc3b8
# ╟─fcca27ba-d4a4-11ea-213a-c3e2305869f1
# ╟─2baaff10-d56c-11ea-2a23-bfa3a7ae2e4b
# ╟─102fbf2e-d56b-11ea-189d-c78d56c0a924
# ╟─cc0d5622-d788-11ea-19cd-3bf6864d9263
# ╟─7c79975c-d789-11ea-30b1-67ff05418cdb
# ╟─5f1c3f6c-d48b-11ea-3eb0-357fd3ece4fc
# ╟─11ddebfe-d488-11ea-223a-e9403f6ec8de
# ╠═030e592e-d488-11ea-060d-97a3bb6353b7
# ╠═c8d26856-d48a-11ea-3cd3-1124cd172f3a
# ╠═37c4394e-d489-11ea-174c-b13bdddbe741
# ╠═fef54688-d48a-11ea-340b-295b88d21382
# ╟─259a2852-d48c-11ea-0f01-b9634850e09d
# ╟─f06fb004-d79f-11ea-0d60-8151019bf8c7
# ╠═26a8a42c-d7a1-11ea-24a3-45bc6e0674ea
# ╠═399552c4-d7a1-11ea-36bb-ad5ca42043cb
# ╟─4bb19760-d7bf-11ea-12ed-4d9e4efb3482
# ╠═5a8ba8f4-d493-11ea-1839-8ba81f86799d
# ╠═a625a922-d493-11ea-1fe9-bdd4a694cde0
# ╟─4fd20ed2-d7a2-11ea-206e-13799234913f
# ╠═692dfb44-d7a1-11ea-00da-af6550bc0622
# ╠═7e4ee09c-d7a1-11ea-0e56-c1921012bc30
# ╟─4c209bbe-d7b1-11ea-0628-33eb8d664f5b
# ╠═fd44a3d4-d7a4-11ea-24ea-09456ff2c53d
# ╟─84272664-d7b7-11ea-2e37-dffd2023d8d6
# ╠═900e2ea4-d7b8-11ea-3511-6f12d95e638a
# ╟─d76be888-d7b4-11ea-2989-2174682ead76
# ╟─8160f4a2-d789-11ea-28f8-e91d58a61642
# ╟─02fb8e62-d4a3-11ea-2a6e-bd415591c891
# ╟─6097b916-d92c-11ea-0dee-e9791b041b67
# ╟─4a3f8c7c-d7bd-11ea-2370-3d6b629bc653
# ╟─737b7440-d4a3-11ea-35ee-27a2b1b2ee35
# ╟─b44e12b8-d4a3-11ea-3f55-776476cd7d69
# ╟─85c9edcc-d789-11ea-14c8-71697cd6a047
# ╟─89de0b7e-d4a2-11ea-278a-a392b1649486
# ╟─bc98ba6e-d7cc-11ea-0f63-79a720e8aa6c
# ╟─8a4d7fba-d789-11ea-2b3b-23f2c4e5cbdf
