import os

scripts_dir = "scripts"
for f in os.listdir(scripts_dir):
    if f.endswith(".py"):
        path = os.path.join(scripts_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()
            
            new_lines = []
            changed = False
            for line in lines:
                if "plt.show()" in line and not line.strip().startswith("#"):
                    new_lines.append(line.replace("plt.show()", "# plt.show()"))
                    changed = True
                else:
                    new_lines.append(line)
            
            if changed:
                with open(path, "w", encoding="utf-8") as f_out:
                    f_out.writelines(new_lines)
                print(f"Disabled plt.show() in {f}")
        except Exception as e:
            print(f"Could not process {f}: {e}")
