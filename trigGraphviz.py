import torch
import matplotlib.pyplot as plt
import numpy as np


def displaySin(dispDer):
    x = torch.linspace(0, 2*np.pi, 100, requires_grad=True)
    siny = torch.sin(x)
    y = torch.sum(siny)
    y.backward()
    if dispDer==1:
        plt.plot(x.detach().numpy(), torch.sin(x).detach().numpy(), label="sin(x)")
        plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="derivative")
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=sin(x) and its derivative")
        
        plt.legend()
        
        plt.show()
    else:
        plt.plot(x.detach().numpy(), torch.sin(x).detach().numpy(), label="sin(x)")
      
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=sin(x)")
        
        plt.legend()
        
        plt.show()
        
    

#cos function:
def displayCos(dispDer):
    x = torch.linspace(0, 2*np.pi, 100, requires_grad=True)
    cosy = torch.cos(x)
    y2 = torch.sum(cosy)
    y2.backward()
    if dispDer ==1:
        plt.plot(x.detach().numpy(), torch.cos(x).detach().numpy(), label="cos(x)")
        plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="derivative")
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=cos(x) and its derivative")
        
        plt.legend()
        plt.show()
    else:
        plt.plot(x.detach().numpy(), torch.cos(x).detach().numpy(), label="cos(x)")
        
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=cos(x)")
        
        plt.legend()
        plt.show()
        
#for tan:
def displayTan(dispDer):
    x = torch.linspace(0, 2*np.pi, 1000, requires_grad=True)
    tany = torch.tan(x)
    safe = (torch.abs(torch.cos(x)) > 0.05)
    y3 = torch.sum(tany[safe])
    y3.backward()
    
    mask=np.abs(torch.cos(x).detach().numpy()) < 0.05
    tan_np = tany.detach().numpy()
    x_np= x.grad.detach().numpy()
    tan_np[mask] = np.nan
    x_np[mask] = np.nan
    if dispDer==1:
        plt.plot(x.detach().numpy(), tan_np, label="tan(x)")
        plt.plot(x.detach().numpy(), x_np, label="derivative")
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=tan(x) and its derivative")
        
        plt.legend()
        plt.grid()
        plt.ylim(-10, 10)
        plt.show()
    else:
        plt.plot(x.detach().numpy(), tan_np, label="tan(x)")
        
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=tan(x)")
        
        plt.legend()
        plt.grid()
        plt.ylim(-10, 10)
        plt.show()
        
    
    
# for sec:
def displaySec(dispDer):
    
    x = torch.linspace(0, 2*np.pi, 1000, requires_grad=True)
    secy = 1/torch.cos(x)
    safe = (torch.abs(torch.sin(x)) > 0.05)
    
    y3 = torch.sum(secy[safe])
    y3.backward()
    x.grad.detach().numpy()[np.abs(torch.cos(x).detach().numpy()) < 0.05] = np.nan
    
    secy[np.abs(torch.cos(x).detach().numpy()) < 0.05] = np.nan

    if dispDer==1:
        
        plt.plot(x.detach().numpy()[safe], secy.detach().numpy()[safe], label="sec(x)")
        plt.plot(x.detach().numpy()[safe], x.grad.detach().numpy()[safe], label="derivative")
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=sec(x) and its derivative")
        plt.grid()
        plt.legend()
        plt.ylim(-10, 10)
        plt.show()
    else:
        plt.plot(x.detach().numpy()[safe], secy.detach().numpy()[safe], label="sec(x)")

        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=sec(x)")
        plt.grid()
        plt.legend()
        plt.ylim(-10, 10)
        plt.show()

#for cosec:
def displayCosec(dispDer):
    
    x = torch.linspace(0, 2*np.pi, 1000, requires_grad=True)
    cosecy = 1 / torch.sin(x)
    
    safe = (torch.abs(torch.sin(x)) > 0.05)
    
    y3 = torch.sum(cosecy[safe])
    y3.backward()
    x.grad.detach().numpy()[np.abs(torch.sin(x).detach().numpy()) < 0.05] = np.nan
    cosecy[np.abs(torch.sin(x).detach().numpy()) < 0.05] = np.nan
    
    if dispDer==1:
        
    
    
        plt.plot(x.detach().numpy(), cosecy.detach().numpy(), label="cosec(x)")
        plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="derivative")
        
        positions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π'])
        plt.title("y = cosec(x) and its derivative")
        plt.legend()
        plt.grid(True)
        plt.ylim(-10, 10) 
        plt.show()
    else:
        plt.plot(x.detach().numpy(), cosecy.detach().numpy(), label="cosec(x)")

        
        positions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π'])
        plt.title("y = cosec(x)")
        plt.legend()
        plt.grid(True)
        plt.ylim(-10, 10) 
        plt.show()

#cot 
def displayCot(dispDer):
    
    x = torch.linspace(0, 2*np.pi, 1000, requires_grad=True)
    tany = torch.tan(x)
    coty = 1/tany
    safe = (torch.abs(torch.sin(x)) > 0.05)
    y3 = torch.sum(coty[safe])
    y3.backward()
    
    mask=np.abs(torch.sin(x).detach().numpy()) < 0.05
    cot_np = coty.detach().numpy()
    x_np= x.grad.detach().numpy()
    cot_np[mask] = np.nan
    x_np[mask] = np.nan

    if dispDer==1:
        
    
        plt.plot(x.detach().numpy(), cot_np, label="cot(x)")
        plt.plot(x.detach().numpy(), x_np, label="derivative")
        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=cot(x) and its derivative")
        
        plt.legend()
        plt.grid()
        plt.ylim(-10, 10)
        plt.show()
    else:
        plt.plot(x.detach().numpy(), cot_np, label="cot(x)")

        positions= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
        plt.xticks(positions, ['0', 'π/4','π/2', '3π/4', 'π','5π/4', '3π/2', '7π/4', '2π' ])
        plt.title("y=cot(x)")
        
        plt.legend()
        plt.grid()
        plt.ylim(-10, 10)
        plt.show()
    

i=0
while True:
    
    i = int(input("""Welcome.\nPress the following digits to view their respective trigonometric function graph or enter 6 to exit:
                    0-sin(x)
                    1-cos(x)
                    2-tan(x)
                    3-cot(x)
                    4-sec(x)
                    5-cosec(x)\n"""))
    if i!=6:
        
        dispDer=int(input("Do you want to display derivative of the function?\n 1-Yes\n 0-No"))
    else:
        break
    if i==6:
        break
    elif i==0:
        displaySin(dispDer)
    elif i==1:
        displayCos(dispDer)
    elif i==2:
        displayTan(dispDer)
    elif i==3:
        displayCot(dispDer)
    elif i==4:
        displaySec(dispDer)
    elif i==5:
        displayCosec(dispDer)
    
            
    else:
        
        print("Invalid input")
        
    
