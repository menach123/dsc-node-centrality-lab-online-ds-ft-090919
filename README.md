
# Network Dynamics: Node Centrality - Lab

## Introduction

In this lab, you'll get a chance to practice implementing and interpreting the centrality metrics from the previous section. You'll do this be investigating the social network from Game of Thrones!

## Objectives
You will be able to:
- Understand and explain network centrality and its importance in graph analysis
- Understand and calculate Degree, Closeness, Betweenness and Eigenvector centrality measures
- Describe the use case for several centrality measures

## Character Interaction Graph Data

A. J. Beveridge, and J. Shan  created a network from George R. Martin's "A song of ice and fire" by extracting relationships between characters of the story. [The dataset is available at Github](https://github.com/mathbeveridge/asoiaf). Relationships between characters were formed every time a character's name appears within 15 words of another character. This was designed as an approximate metric for character's interactions with each other. The results of this simple analysis are quite profound and produce interesting visuals such as this graph:

<img src="images/got.png" width=800>

With that, it's your turn to start investigating the most central characters!


```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')

```

##  Load the dataset 

Start by loading the dataset as a pandas DataFrame. From this, you'll then create a network representation of the dataset using NetworkX. 

The dataset is stored in the file `asoiaf-all-edges.csv`.


```python
df = pd.read_csv('asoiaf-all-edges.csv')
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Target</th>
      <th>Type</th>
      <th>id</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Addam-Marbrand</td>
      <td>Brynden-Tully</td>
      <td>Undirected</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Addam-Marbrand</td>
      <td>Cersei-Lannister</td>
      <td>Undirected</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Addam-Marbrand</td>
      <td>Gyles-Rosby</td>
      <td>Undirected</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Addam-Marbrand</td>
      <td>Jaime-Lannister</td>
      <td>Undirected</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Addam-Marbrand</td>
      <td>Jalabhar-Xho</td>
      <td>Undirected</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
nodes = []
for i in df.Target.unique():
    nodes.append(i)
for i in df.Source.unique():
    nodes.append(i)

len(set(nodes))
```




    796



## Create a Graph

Now that you have the data loaded as a pandas DataFrame, iterate through the data and create appropriate edges to the empty graph you instantiated above. Be sure to add the weight to each edge.


```python
# Create an empty graph instance
G = nx.Graph()

# Read edge lists into dataframes
for node in nodes:
    G.add_node(node)

for dict_ in df.to_dict(orient='records'):
    G.add_edge(dict_['Source'], dict_['Target'], weight = dict_['weight'])
```

## Calculate Degree

To start the investigation of the most central characters in the books, calculate the degree centrality for each character. Then create a bar graph of the top 10 characters according to degree centrality.


```python
plt.figure(figsize=(13,5))
pd.Series(dict(nx.degree(G))).sort_values(ascending=False)[:10].plot.bar(fontsize=12)
plt.title('Characters with the Most Centrality')
plt.show()
```


![png](index_files/index_9_0.png)


## Closeness Centrality

Repeat the above exercise for the top 10 characters according to closeness centrality.


```python
plt.figure(figsize=(13,5))
pd.Series(dict(nx.closeness_centrality(G))).sort_values(ascending=False)[:10].plot.bar(fontsize=14)
plt.title('Characters with the Closeness Centrality')
plt.show()
```


![png](index_files/index_11_0.png)


## Betweeness Centrality

Repeat the process one more time for betweeness centrality.


```python
plt.figure(figsize=(13,5))
pd.Series(dict(nx.betweenness_centrality(G))).sort_values(ascending=False)[:10].plot.bar(fontsize=14)
plt.title('Characters with the Betweeness Centrality')
plt.show()

```


![png](index_files/index_13_0.png)


## Putting it All Together

Great! Now try putting all of these metrics together along with eigenvector centrality. Combine all four metrics into a single dataframe for each character.


```python
plt.figure(figsize=(13,5))
pd.Series(dict(nx.eigenvector_centrality(G))).sort_values(ascending=False)[:10].plot.bar(fontsize=14)
plt.title('Characters with the Eigenvector Centrality')
plt.show()


```


![png](index_files/index_15_0.png)



```python
character_df = pd.DataFrame()
character_df['degree'] = pd.Series(dict(nx.degree(G)), name='degree')
character_df['closeness'] = pd.Series(dict(nx.closeness_centrality(G)))
character_df['betweenness'] = pd.Series(dict(nx.betweenness_centrality(G)))
character_df['eigenvector'] = pd.Series(dict(nx.eigenvector_centrality(G)))
```


```python
character_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree</th>
      <th>closeness</th>
      <th>betweenness</th>
      <th>eigenvector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Brynden-Tully</th>
      <td>19</td>
      <td>0.371495</td>
      <td>0.002227</td>
      <td>0.060019</td>
    </tr>
    <tr>
      <th>Cersei-Lannister</th>
      <td>97</td>
      <td>0.454545</td>
      <td>0.088704</td>
      <td>0.235771</td>
    </tr>
    <tr>
      <th>Gyles-Rosby</th>
      <td>18</td>
      <td>0.339453</td>
      <td>0.000415</td>
      <td>0.059528</td>
    </tr>
    <tr>
      <th>Jaime-Lannister</th>
      <td>101</td>
      <td>0.451961</td>
      <td>0.100838</td>
      <td>0.226339</td>
    </tr>
    <tr>
      <th>Jalabhar-Xho</th>
      <td>5</td>
      <td>0.313733</td>
      <td>0.000807</td>
      <td>0.012674</td>
    </tr>
  </tbody>
</table>
</div>



## Identifying Key Players

While centrality can tell us a lot, you've also begun to see how certain individuals may not be the most central characters, but can be pivotal in the flow of information from one community to another. In the previous lesson, such nodes were labeled as 'bridges' acting as the intermediaries between two clusters. Try and identify such characters from this dataset.


```python
character_df.loc[(character_df.degree < character_df.degree.quantile(.99))].sort_values('betweenness', ascending=False).iloc[:5]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree</th>
      <th>closeness</th>
      <th>betweenness</th>
      <th>eigenvector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Daenerys-Targaryen</th>
      <td>73</td>
      <td>0.383317</td>
      <td>0.118418</td>
      <td>0.063043</td>
    </tr>
    <tr>
      <th>Theon-Greyjoy</th>
      <td>66</td>
      <td>0.423323</td>
      <td>0.111283</td>
      <td>0.102481</td>
    </tr>
    <tr>
      <th>Eddard-Stark</th>
      <td>74</td>
      <td>0.455849</td>
      <td>0.078732</td>
      <td>0.191660</td>
    </tr>
    <tr>
      <th>Robert-Baratheon</th>
      <td>65</td>
      <td>0.459272</td>
      <td>0.078228</td>
      <td>0.194375</td>
    </tr>
    <tr>
      <th>Robb-Stark</th>
      <td>74</td>
      <td>0.444134</td>
      <td>0.066468</td>
      <td>0.173196</td>
    </tr>
  </tbody>
</table>
</div>



## Drawing the Graph

To visualize all of these relationships, draw a graph of the network.


```python
fig = plt.figure(figsize=(15,10))
#Draw the network!
nx.draw(G, pos=nx.spring_layout(G), with_labels=True,
        alpha=.8, node_color="#1cf0c7", node_size=700)
```


![png](index_files/index_21_0.png)


## Subsetting the Graph

As you can see, the above graph is undoubtedly noisy, making it difficult to discern any useful patterns. As such, reset the graph and only add edges whose weight is 75 or greater. From there, redraw the graph. To further help with the display, try using `nx.spring_layout(G)` for the position. To jazz it up, try and recolor those nodes which you identified as bridge or bottleneck nodes to communication.


```python
G_ = nx.convert_matrix.from_pandas_edgelist(df.loc[df.weight>50], 'Source', 'Target', ['weight'])
fig = plt.figure(figsize=(15,10))
#Draw the network!
nx.draw(G_ , pos=nx.spring_layout(G_ ), with_labels=True,
        alpha=.8)
```


![png](index_files/index_23_0.png)


## Summary 

In this lab, we looked at different centrality measures of the graph data for the ASIOF dataset. We also compared these measures to see how they correlate with each other. We also saw in practice, the difference between taking the weighted centrality measures and how it may effect the results. 
