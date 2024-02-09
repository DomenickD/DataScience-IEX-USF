#Currency Conversion App Stormwind

#enter local currency
#enter currency to convert to
#enter the exchange

#import modules
import tkinter as tk
from tkinter import *
from tkinter import ttk #tabs notebooks
from tkinter import messagebox

window = tk.Tk() #to create a window
window.title( "Simple Currency Converter" ) 
window.geometry( '500x500' ) #set size of window

#create tabs
my_tab = ttk.Notebook(window)
my_tab.pack(pady=5, padx=5)

#create 2 tabs
currency_frame = Frame( my_tab, width=480, height=480 )
conversion_frame = Frame ( my_tab, width=480, height=480 )

#adding frame to tab
my_tab.add(currency_frame, text="Currency")
my_tab.add(conversion_frame, text="Convert")

def lock():
    #check if there is anything listed in the entry boxes
    if not home_entry.get() or not conversion_entry.get or not rate_entry.get():
        messagebox.showwarning("WARNING!! You did not fill out all the fields.")
    else:
        #disable the entry boxes
        home_entry.config(state='disable')
        conversion_entry.config(state='disable')
        rate_entry.config(state='disable')

    #enable convert tab
    my_tab.tab(1, state="normal")

    #changing tab field
    amount_label.config(text=f"Amount of {home_entry.get()} To Convert to {conversion_entry.get()}")
    converted_label.config(text= f"Equals This Many {conversion_entry.get()}")
    convert_button.config(text= f"Convert from {home_entry.get()}")

def unlock():
    #enable entry boxes
    home_entry.config(state="normal")
    conversion_entry.config(state="normal")
    rate_entry.config(state="normal")

    #disable conversion tab
    my_tab.tab(1, state="disable")



#local currency entry box
home = LabelFrame(currency_frame, text= "Enter Currency to Convert:")
home.pack(pady=20, padx=20)

home_entry = Entry(home, font = ("Calibri", 25))
home_entry.pack(pady=10, padx=10)

#creating conversion LabalFrame
conversion = LabelFrame(currency_frame, text="Conversion")
conversion.pack(pady=20)

#creating conversion currency label
conversion_label = Label(conversion, text="Enter Currency to Convert to: ")
conversion_label.pack(pady=10)

#create conversion currency entry box
conversion_entry = Entry(conversion, font = ("Calibri", 25))
conversion_entry.pack(padx=10, pady=10)

#creating exchange rate labe;
rate_label =  Label(conversion, text="Current Exchange/Conversion Rate: ")
rate_label.pack(pady=10)

#creating entry box for rate label
rate_entry = Entry(conversion, font= ("Calibri", 25))
rate_entry.pack(pady=10, padx=10)

#creating button frame
button_frame = Frame(currency_frame)
button_frame.pack(pady=20)

#creating buttons 
lock_button = Button(button_frame, text="Lock", command=lock)
lock_button.grid(row=0, column=0, padx=10)

unlock_button = Button(button_frame, text="Unlock", command=unlock)
unlock_button.grid(row=0, column=1, padx=10)

#######
# CONVERSION TAB
#######

def convert():
    #clear
    converted_entry.delete(0,END)
    
    #get values from the entry boxes and convert them to floats
    conversion = float(rate_entry.get()) * float(amount_entry.get())
    
    #convert to 2 decimal places
    conversion  = round(conversion, 2)

    #adding commas
    conversion = '{:,}'.format(conversion)

    #insert into the converted amount field
    converted_entry.insert(0, f"${conversion}")




def clear():
    amount_entry.delete(0, END)
    converted_entry.delete(0, END)


#create amount label
amount_label = Label(conversion_frame, text="Amount to convert")
amount_label.pack(pady=20)

#creare enrtry amount entry box
amount_entry = Entry(amount_label, font = ("Calibri", 25))
amount_entry.pack(pady=10, padx=10)

#creating convert button
convert_button = Button(amount_label, text="Convert", command=convert)
convert_button.pack(pady=20)

#output label_Frame
converted_label = LabelFrame(conversion_frame, text= "Converted Currency")
converted_label.pack(pady=10)

#creating converted entry
converted_entry = Entry(converted_label, font = ("Calibri", 25), bd=0, bg="grey")
converted_entry.pack(pady=10, padx=10)

#clear button
clear_button = Button(conversion_frame, text="clear", command=clear)
clear_button.pack(pady=20)


window.mainloop()