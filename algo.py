from gerrychain import Graph, Partition, Election
from gerrychain.updaters import Tally, cut_edges

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, within_percent_of_ideal_population
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

import pandas as pd

import matplotlib.pyplot as plt

def main():
    # subgraph = Graph.from_file("tl_2020_36001_tabblock20.zip")
    # for node in subgraph:
    #     print(subgraph.nodes[node].items())
    #     return

    graph = Graph.from_file("./tl_2020_36_vtd20.zip")
    graph.add_edge(10308, 10677)
    print(graph.islands)

    print(graph.number_of_nodes)

    df = pd.read_csv("ny_2020_vtd.csv")
    df["GEOID20vtd"] = df["GEOID20vtd"].astype(str)
    graph.join(df, None, "GEOID20", "GEOID20vtd")

    df = pd.read_csv("ny_cong_2022_invalidated.csv")
    df["GEOID20"] = df["GEOID20"].astype(str)
    graph.join(df, None, "GEOID20", "GEOID20")

    election = Election("CLN20", {"Dem": "adv_20", "Rep": "arv_20"})

    initial_partition = Partition(
        graph,
        assignment="District",
        updaters={
            "cut_edges": cut_edges,
            "population": Tally("pop", alias="population"),
            "CLN20": election
        }
    )

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[single_flip_contiguous],
        accept=always_accept,
        initial_state=initial_partition,
        total_steps=10000
    )

    d_percents = [sorted(partition["CLN20"].percents("Dem")) for partition in chain]

    data = pd.DataFrame(d_percents)

    data.to_csv("single_flip.csv")
    
    assignment_dict = chain.state.assignment
    districts = pd.DataFrame(data=[(graph.nodes[k]["GEOID20"], assignment_dict[k]) for (k, _) in enumerate(assignment_dict)], columns=["GEOID20", "District"])
    districts.to_csv("district_assignments.csv")

    ax = data.boxplot(positions=range(len(data.columns)))
    plt.plot(data.iloc[0], "ro")

    plt.show()

if __name__=="__main__":
    main()