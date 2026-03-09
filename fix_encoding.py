import os

mapping = {
    "\u2248": "approx.",
    "\u2192": "->",
    "\u03bb": "lambda",
    "\u03c1": "rho",
    "\u03c3": "sigma",
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03b5": "epsilon",
    "\u03b6": "zeta",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\u221e": "inf",
    "\u2211": "sum",
    "\u2202": "partial",
    "\u2207": "grad",
    "\u2212": "-",
    "\u2215": "/",
    "\u2217": "*",
    "\u2219": "*",
    "\u221a": "sqrt",
    "\u221d": "prop",
    "\u2220": "angle",
    "\u2227": "and",
    "\u2228": "or",
    "\u2229": "cap",
    "\u222a": "cup",
    "\u222b": "int",
    "\u2234": "therefore",
    "\u2235": "because",
    "\u2236": ":",
    "\u2237": "::",
    "\u2238": "-.",
    "\u2239": "-:",
    "\u223a": ":-",
    "\u223b": ":-:",
    "\u223c": "~",
    "\u223d": "~",
    "\u223e": "~",
    "\u223f": "~",
    "\u2240": "~",
    "\u2241": "!~",
    "\u2242": "=",
    "\u2243": "~=",
    "\u2244": "!~=",
    "\u2245": "~~",
    "\u2246": "!~~",
    "\u2247": "~~",
    "\u2248": "~~",
    "\u03bc": "mu",
    "\u03c0": "pi",
    "\u03c9": "omega",
    "\u0394": "Delta",
    "\u0398": "Theta"
}

scripts_dir = "scripts"
for f in os.listdir(scripts_dir):
    if f.endswith(".py"):
        path = os.path.join(scripts_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as f_in:
                content = f_in.read()
            
            new_content = content
            # Add a fix for specific problematic strings if needed
            new_content = new_content.replace("\u2192", "->")
            new_content = new_content.replace("\u2248", "approx.")
            new_content = new_content.replace("\u03bb", "lambda")
            new_content = new_content.replace("\u03c1", "rho")
            
            # More general replacements for any mathematical symbols that might break console
            for k, v in mapping.items():
                new_content = new_content.replace(k, v)
                
            if new_content != content:
                with open(path, "w", encoding="utf-8") as f_out:
                    f_out.write(new_content)
                print(f"Fixed {f}")
        except Exception as e:
            print(f"Could not fix {f}: {e}")
