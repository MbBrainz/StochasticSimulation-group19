# Notes Code Harshita

## implementation notes



## TO DO
- if we want to properly test the search algo we need to isolate the simulation fumnction and test it.
- write automated simullation function with data store
- add statistical hypothesis test to test function
- generalize datastorage (dataframe)

## general notes
- i placed(local) the datafile in the repo, because its only 20 kB
- As we have the theoretical value, we can apply statistical tests, YAY


## Questions for
- If we choose Tinitial, cooling factor and markov chains, is that enough to get a nice grade?
- How do we know if it is a good local minima? Is showing that it is close to the global minimum enough of proof or do we need to do something else?
- add statistical hypothesis test to test function, is that nessecary? Hypothesis test on the found minimum possible

## Answers:
- Tinitial may also be found in literature but we can do both
- local minimum is only statistical estimate, cant be sure.
- to proof effectiveness of the algorithm we need to provide statistical signifuicance for multiple iterations (std, CI), but we can't really do a test because the theoretical value will asways be lower. Stick to percentage errors and check how other papers do it.

## todo maupi
- [writing]
    - methods: Simulated Anhealing compared to litrature and other approaches
    - Mutation: BR versus Brute force

- [questions] send answers to Harshita
    - search some stuf about "benchmarking"

- [code]
    - isolate the simulation fumnction and test it. [DONE]
    - generalize datastorage (dataframe)
    - write automated simullation function with data store
    - distance_matrix calculation is using loops, probably not fast due to that.
    - total_distance same