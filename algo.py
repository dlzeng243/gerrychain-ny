from re import S
from gerrychain import Graph, Partition, Election, MarkovChain, constraints
from gerrychain.updaters import Tally, cut_edges

from gerrychain.proposals import recom
from gerrychain.accept import always_accept

from functools import partial

import pandas as pd

import matplotlib.pyplot as plt

names = ['ny_sldu_2022_invalidated_precincts', 'statesenate_ccny_precincts', 'wilson_prieve_senate_precincts']

def main():
    # proposal to analyze
    map_name = 'ari_sponza_precincts'
    for map_name in names:
        print("Running", map_name)

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
        df = pd.read_csv(f"precincts/{map_name}.csv")
        df["GEOID20"] = df["GEOID20"].astype(str)
        graph.join(df, None, "GEOID20", "GEOID20")

        # initialize the MCMC algorithm
        # election is essentially looking at the average democratic votes and average republican votes over 2020
        election = Election("CLN20", {"Dem": "adv_20", "Rep": "arv_20"})
        # initial partition takes in the graph and assignment above, and the updaters are the following:
            # cut_edges: edges of precincts that are between two districts
            # population: updates population
            # CLN20: the election results
        initial_partition = Partition(
            graph,
            assignment="District",
            updaters={
                "cut_edges": cut_edges,
                "population": Tally("pop", alias="population"),
                "CLN20": election
            }
        )


        # senate - used 0.05 population bound and 75 max split counties for ccny_senate
        # congress - used 0.02 population bound and 50 max split counties

        SENATE_EPS = 0.05
        CONG_EPS = 0.02
        SENATE_SPLIT = 75
        CONG_SPLIT = 50

        # ideal population is to make sure the population stays within a certain bound of being equal among the districts
        # we are only looking at precincts and no blocks, so population varies a lot more
        ideal_population = sum(initial_partition['population'].values())/len(initial_partition)
        proposal = partial(recom, pop_col='pop', pop_target=ideal_population, epsilon=SENATE_EPS, node_repeats=2)
        # compactness bound - makes sure we don't make too many cut edges
        compactness_bound = constraints.UpperBound(
            lambda p: len(p["cut_edges"]),
            2*len(initial_partition["cut_edges"])
        )
        # population constraint
        pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, SENATE_EPS)

        # splitting constraint - counts number of counties split and counts number of times counties were split
        def check_split(edge):
            if graph.nodes[edge[0]]["county"] == graph.nodes[edge[1]]["county"]:
                return graph.nodes[edge[0]]["county"]
            return None
        def check_split_county(p):
            s = set()
            d = set()
            for edge in p["cut_edges"]:
                if graph.nodes[edge[0]]["county"] == graph.nodes[edge[1]]["county"]:
                    s.add((graph.nodes[edge[0]]["county"], p.assignment[edge[0]]))
                    s.add((graph.nodes[edge[0]]["county"], p.assignment[edge[1]]))
                    d.add(graph.nodes[edge[0]]["county"])
            return len(s) - len(d)
        splitting_constraint = constraints.UpperBound(
            lambda p: len(set(map(check_split, p["cut_edges"]))), 35
        )
        splitting_constraint2 = constraints.UpperBound(
            check_split_county, SENATE_SPLIT
        )

        # run the markov chain with the above proposal / constraints
        chain = MarkovChain( 
            proposal=proposal,
            constraints=[compactness_bound, pop_constraint, constraints.contiguous, splitting_constraint, splitting_constraint2],
            accept=always_accept,
            initial_state=initial_partition,
            total_steps=250
        )

        # run the algorithm and output data
        # this data is the sorted percentage of wins for the democrats
        d_percents = [sorted(partition["CLN20"].percents("Dem")) for partition in chain][50:]
        print(len(d_percents))
        data = pd.DataFrame(d_percents)
        data.to_csv(f"district_assignments/{map_name}_simulation_data.csv")
        
        # this data is the resulting precinct to district assignment from running gerrychain
        assignment_dict = chain.state.assignment
        districts = pd.DataFrame(data=[(graph.nodes[k]["GEOID20"], assignment_dict[k]) for k, _ in enumerate(assignment_dict)], columns=["GEOID20", "District"])
        districts.to_csv(f"district_assignments/{map_name}_district_assignments.csv", index=False)

        # the following is to make a plot of the win percentages
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

        plt.savefig(f"district_assignments/{map_name}_plot.png", bbox_inches='tight')
        plt.clf()
    #plt.show()

if __name__=="__main__":
    main()