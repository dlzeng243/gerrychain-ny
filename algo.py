from gerrychain import Graph, Partition, Election, MarkovChain, constraints
from gerrychain.updaters import Tally, cut_edges

from gerrychain.proposals import recom
from gerrychain.accept import always_accept

from functools import partial

import pandas as pd

import matplotlib.pyplot as plt

def main():

    # initialize the graph
    graph = Graph.from_file("./map_data/tl_2020_36_vtd20.zip")
    graph.add_edge(10308, 10677)
    print(graph.islands) # should have no islands
    print(graph.number_of_nodes)

    # import voter information (from 2020)
    df = pd.read_csv("./map_data/ny_2020_vtd.csv")
    df["GEOID20vtd"] = df["GEOID20vtd"].astype(str)
    graph.join(df, None, "GEOID20", "GEOID20vtd")

    # import initial district assignments
    df = pd.read_csv("ny_cong_2022_invalidated.csv")
    df["GEOID20"] = df["GEOID20"].astype(str)
    graph.join(df, None, "GEOID20", "GEOID20")

    # initialize the MCMC algorithm
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

    ideal_population = sum(initial_partition['population'].values())/len(initial_partition)
    proposal = partial(recom, pop_col='pop', pop_target=ideal_population, epsilon=0.0075, node_repeats=2)
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]),
        2*len(initial_partition["cut_edges"])
    )
    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.0075)

    chain = MarkovChain(
        proposal=proposal,
        constraints=[compactness_bound, pop_constraint, constraints.contiguous],
        accept=always_accept,
        initial_state=initial_partition,
        total_steps=100
    )

    # run the algorithm and output data
    d_percents = [sorted(partition["CLN20"].percents("Dem")) for partition in chain]
    data = pd.DataFrame(d_percents)
    data.to_csv("single_flip.csv")
    
    assignment_dict = chain.state.assignment
    districts = pd.DataFrame(data=[(graph.nodes[k]["GEOID20"], assignment_dict[k]) for k, _ in enumerate(assignment_dict)], columns=["GEOID20", "District"])
    districts.to_csv("district_assignments.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw 50% line
    ax.axhline(0.5, color="#cccccc")

    # Draw boxplot
    data.boxplot(ax=ax, positions=range(len(data.columns)))

    # Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
    plt.plot(data.iloc[0], "ro")

    # Annotate
    ax.set_title("Comparing the 2022 NY Congressional plan to an ensemble")
    ax.set_ylabel("Democratic vote % (averaged 2020 elections)")
    ax.set_xlabel("Sorted districts")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    plt.show()

if __name__=="__main__":
    main()