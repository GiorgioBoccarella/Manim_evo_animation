# Evolutionary theory animation with manim

I built these simulations from scratch and visualized them with an engine for programmatic animation. If you want to reproduce the work shown here you should first set up your [manim community edition](https://github.com/ManimCommunity/manim/) installation.


# Fitness landscapes

### Note that in the previews the mutation rate and mutation size are quite large to showcase a short preview simulation


## Usage

The simulations paramters are stored in a dictionary in common.py:

```python
params_sim = {
    "res": 3,
    "seed": 124,
    "max_gen": 90,
    "pop_size": 150,
    "mut_rate": 0.4,
    "perlin_seed": 12356,
}

```

In order to view the output of a scene (desnity dependent landscape for example), run the following in a terminal window:

```sh
manim -pl -ql density_dependent_landscape.py SimPlot
```

Simulation and graphic parameters specified on file (common.py). 

### **- Preview of Frequency-dependent selection simulation**


![f](https://user-images.githubusercontent.com/45296503/134049588-59e9c0b9-1317-4e6c-b328-45bb33fc94d4.png)


### Video: 

https://user-images.githubusercontent.com/45296503/135101101-44bac0ca-6a63-44d1-be5c-7766c05d35b3.mp4



### **- Preview of selection on static landscape**

![s](https://user-images.githubusercontent.com/45296503/134049673-300dfb29-4779-4259-8b54-c6ecd876fa5f.png)



### Video: 
https://user-images.githubusercontent.com/45296503/134049688-4eb328d6-096f-47ce-807f-a02fb3976d54.mp4



### **- Preview of selection on dynamic landscape**
![d](https://user-images.githubusercontent.com/45296503/134049706-5d4b5971-b972-463a-ad0f-abcd6cd0ea80.png)




### Video: 
https://user-images.githubusercontent.com/45296503/134049788-8c6f9907-1175-4529-84d4-91237c7e46d7.mp4



# **Trajectory along a "Neutral genotype Network"**


Animation inspired by *"The origin of evolutionary innovation" Chapter 5 (Andreas Wagner)*.

Different connectedness values favour the discovery on novel phenotypes in the genotype network 

![image](https://user-images.githubusercontent.com/45296503/132538529-848b9e73-49eb-4b38-b77f-e72cce08c7ae.png)


https://user-images.githubusercontent.com/45296503/132538554-b39ad2e4-7a95-4676-8ad8-accdaa9a2323.mp4




# **Network modularity**

Generate modular networks with certain features by rewiring edges between clusters. Modules are generated via networkx and modularity is calculated with a built-in function based on the Newman algorithm. 

![ModTest_ManimCE_v0 9 0](https://user-images.githubusercontent.com/45296503/129604123-0823977f-ee07-467f-b866-93a2b9f79055.png)



https://user-images.githubusercontent.com/45296503/135896335-bd6d0209-4ef0-45b5-8b0c-2150bd4689ed.mp4


