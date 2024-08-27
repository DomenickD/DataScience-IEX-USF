import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


def update_details(*args):
    command = command_var.get()
    detail = detail_var.get()
    detail_text.delete("1.0", tk.END)

    if command == "Add":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command adds the specified files to the staging area for the next commit.",
        )
    elif command == "Commit":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command commits the staged changes with a message.",
        )
    elif command == "Push":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command pushes the committed changes to the remote repository.",
        )
    elif command == "Pull":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command fetches changes from the remote repository and merges them into the current branch.",
        )
    elif command == "Branch":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command creates, lists, renames, or deletes branches.",
        )
    elif command == "Checkout":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command switches to the specified branch or restores working tree files.",
        )
    elif command == "Merge":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command merges the specified branch into the current branch.",
        )
    elif command == "Clone":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command clones a repository into a new directory.",
        )
    elif command == "Fetch":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command fetches changes from the remote repository without merging them.",
        )
    elif command == "Status":
        detail_text.insert(
            tk.END,
            f"Command: git {command}\nThis command displays the state of the working directory and the staging area.",
        )
    elif command == "Log":
        detail_text.insert(
            tk.END, f"Command: git {command}\nThis command shows the commit logs."
        )


def update_output():
    command = command_var.get()
    detail = detail_var.get()
    output = ""

    if command == "Add":
        output = f"git add {detail}"
    elif command == "Commit":
        output = f'git commit -m "{detail}"'
    elif command == "Push":
        output = f'git add *\ngit commit -m "{detail}"\ngit push origin main'
    elif command == "Pull":
        output = "git pull"
    elif command == "Branch":
        output = f"git branch {detail}"
    elif command == "Checkout":
        output = f"git checkout {detail}"
    elif command == "Merge":
        output = f"git merge {detail}"
    elif command == "Clone":
        output = f"git clone {detail}"
    elif command == "Fetch":
        output = "git fetch"
    elif command == "Status":
        output = "git status"
    elif command == "Log":
        output = "git log"

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, output)


def clear_fields():
    command_var.set("Select Command")
    detail_var.set("")
    detail_text.delete("1.0", tk.END)
    output_text.delete("1.0", tk.END)


root = tk.Tk()
root.title("Git Command Helper")

tab_control = ttk.Notebook(root)

command_tab = ttk.Frame(tab_control)
detail_tab = ttk.Frame(tab_control)
output_tab = ttk.Frame(tab_control)

tab_control.add(command_tab, text="Command")
tab_control.add(detail_tab, text="Details")
tab_control.add(output_tab, text="Output")

tab_control.pack(expand=1, fill="both")

# Command Tab
command_var = tk.StringVar(value="Select Command")
commands = [
    "Add",
    "Commit",
    "Push",
    "Pull",
    "Branch",
    "Checkout",
    "Merge",
    "Clone",
    "Fetch",
    "Status",
    "Log",
]
command_label = ttk.Label(command_tab, text="Select Git Command:")
command_label.pack(pady=10)
command_combo = ttk.Combobox(command_tab, textvariable=command_var, values=commands)
command_combo.pack(pady=10)
command_var.trace("w", update_details)

# Detail Tab
detail_var = tk.StringVar()
detail_label = ttk.Label(detail_tab, text="Enter Details:")
detail_label.pack(pady=10)
detail_entry = ttk.Entry(detail_tab, textvariable=detail_var)
detail_entry.pack(pady=10)
detail_text = tk.Text(detail_tab, height=10, width=50)
detail_text.pack(pady=10)

# Output Tab
output_text = tk.Text(output_tab, height=10, width=50)
output_text.pack(pady=10)

# Buttons
button_frame = ttk.Frame(root)
button_frame.pack(fill="x", padx=10, pady=10)

update_button = ttk.Button(button_frame, text="Update Output", command=update_output)
update_button.pack(side="left", padx=10)

clear_button = ttk.Button(button_frame, text="Clear", command=clear_fields)
clear_button.pack(side="right", padx=10)

root.mainloop()
