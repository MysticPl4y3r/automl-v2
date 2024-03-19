from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple
import pandas as pd
from functools import partial
from PreProcessor import PreProcessor
import numpy as np

Genome = dict["seq":List[int],"fitness":int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
TARGET_VAR=str
TOTAL_ENTROPY=float
FitnessFunc = Callable[[Genome,TARGET_VAR], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome,FitnessFunc], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]
Data = pd.DataFrame
Data_Batch=List[pd.DataFrame]

class genetic:
    def __init__(self,Data:pd.DataFrame,targer_var:str) -> None:
        self.Data=Data
        self.Data_Batch=[]
        self.TARGET_VAR=targer_var
        self.TOTAL_ENTROPY=0

    def generate_genome(self,length: int,fitness_func: FitnessFunc) -> Genome:
        temp ={}
        temp['seq']=choices([0, 1], k=length)
        temp['fitness']=0
        temp['fitness']=fitness_func(self,temp)
        return temp

    def generate_population(self,size: int, genome_length: int,fitness_func:FitnessFunc) -> Population:
        return [self.generate_genome(genome_length,fitness_func) for _ in range(size)]

    def single_point_crossover(self,a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        if len(a.get('seq')) != len(b.get('seq')):
            raise ValueError("Genomes a and b must be of same length")

        length = len(a)
        if length < 2:
            return a, b

        p = randint(1, length - 1)
        temp1={}
        temp1['seq']=a['seq'][0:p] + b['seq'][p:]
        temp1['fitness']=0
        temp2={}
        temp2['seq']=b['seq'][0:p] + a['seq'][p:]
        temp2['fitness']=0
        return temp1, temp2

    def mutation(self,genome: Genome,fitness_func:FitnessFunc, num: int = 1, probability: float = 0.5) -> Genome:
        for _ in range(num):
            index = randrange(len(genome))
            genome["seq"][index] = genome["seq"][index] if random() > probability else abs(genome["seq"][index] - 1)
        genome['fitness']=fitness_func(self,genome)
        return genome

    def population_fitness(self,population: Population, fitness_func: FitnessFunc) -> int:
        return sum([genome.get("fitness") for genome in population])

    def selection_pair(self,population: Population, fitness_func: FitnessFunc) -> Population:
        weights=[gene.get("fitness") for gene in population]
        # print(weights)
        return choices(
            population=population,
            weights=weights,
            k=2
        )

    def sort_population(self,population: Population, fitness_func: FitnessFunc) -> Population:
        return sorted(population, key=self.fitness_func, reverse=True)

    def genome_to_string(self,genome: Genome) -> str:
        return "".join(map(str, genome))

    def print_stats(self,population: Population, generation_id: int, fitness_func: FitnessFunc):
        print("GENERATION %02d" % generation_id)
        print("=============")
        # print("Population: [%s]" % ", ".join([self.genome_to_string(gene) for gene in population]))
        print("Avg. Fitness: %f" % (self.population_fitness(population, fitness_func) / len(population)))
        sorted_population = self.sort_population(population, fitness_func)
        print(
            "Best: %s (%f)" % (self.genome_to_string(sorted_population[0]), fitness_func(self,sorted_population[0])))
        print("Worst: %s (%f)" % (self.genome_to_string(sorted_population[-1]),
                                self.fitness_func(sorted_population[-1])))
        print("")

        return sorted_population[0]

    def from_genome(self,genome: Genome, columns: List) -> List:
        new_list = columns.copy()
        new_list.remove(self.TARGET_VAR)
        result = []
        for i, column in enumerate(new_list):
            if genome['seq'][i] == 1:
                result.append(column)
        result.append(self.TARGET_VAR)
        return result

    def entropy(self,class_probs):
        entropy = 0
        for prob in class_probs:
            if prob != 0:
                entropy -= prob * np.log2(prob)
        return entropy
        # class_probs = np.array(class_probs)
        # epsilon = 1e-10  # Small epsilon to avoid log(0)
        # class_probs = np.clip(class_probs, epsilon, 1 - epsilon)  # Clip probabilities to avoid log(0)
        # return -np.sum(class_probs * np.log2(class_probs))

    def information_gain(self,data, target_column):
        # print(data.columns)
        total_info_gain = 0
        num_columns = len(data.columns) - 1  # Exclude target column
        for column in data.columns:
            if column != target_column:
                column_entropy = 0
                unique_values = set(data[column])
                # print(unique_values)
                for value in unique_values:
                    subset = data[data[column] == value]
                    subset_class_probs = [(subset[target_column] == c).mean() for c in set(subset[target_column])]
                    column_entropy += (len(subset) / len(data)) * self.entropy(subset_class_probs)
                total_info_gain += self.TOTAL_ENTROPY - column_entropy
        if num_columns==0:
            return 0
        # average_info_gain = total_info_gain / num_columns
        return total_info_gain
    
    def information_gain2(self,data, target_column):
        # print(data.columns)
        column_entropies =[]
        total_info_gain = 0
        num_columns = len(data.columns) - 1  # Exclude target column
        for column in data.columns:
            if column != target_column:
                column_entropy = 0
                unique_values = set(data[column])
                for value in unique_values:
                    subset = data[data[column] == value]
                    subset_class_probs = [(subset[target_column] == c).mean() for c in set(subset[target_column])]
                    column_entropy += (len(subset) / len(data)) * self.entropy(subset_class_probs)
                total_info_gain += self.TOTAL_ENTROPY - column_entropy
                column_entropies.append(self.TOTAL_ENTROPY - column_entropy)

        if num_columns==0:
            return 0
        total_val=0
        for val in column_entropies:
            total_val+= (val*val)/total_info_gain
        
        return total_val/num_columns

    def get_random_subsets(self,dataframe:pd.DataFrame)->List[pd.DataFrame]:
        total_rows = len(dataframe)
        # Dynamic calculation of batch size
        batch_count=0
        batch_size=0
        if(total_rows>=2000):
            batch_count = total_rows // 2000
            batch_size=2000
        else:
            batch_count=1
            batch_size=total_rows
        subsets = []
        for _ in range(batch_count):
            # Randomly choose rows without replacement
            indices = np.random.choice(total_rows, size=batch_size, replace=False)
            # print(indices)
            subset = dataframe.iloc[indices].copy()  # Make a copy to avoid modifying the original DataFrame
            subsets.append(subset)
        # print(subsets)
        return subsets

    def fitness_func(self,genome:Genome)->int:
        # print(self.Data_Batch)
        if genome.get('fitness')==0:
            index = randint(0,len(self.Data_Batch)-1)
            # index=0
            current_batch = self.Data_Batch[index]
            current_batch = current_batch[self.from_genome(genome,current_batch.columns.to_list())]
            # print(current_batch.columns)
            return self.information_gain(current_batch,self.TARGET_VAR)
        else:
            # print('im called')
            return genome.get('fitness')


    def run_evolution(self,
            popluation_size=int,
            genome_length=int,
            populate_func: PopulateFunc=generate_population,
            fitness_func: FitnessFunc = fitness_func,
            selection_func: SelectionFunc = selection_pair,
            crossover_func: CrossoverFunc = single_point_crossover,
            mutation_func: MutationFunc = mutation,
            generation_limit: int = 100) \
            -> Tuple[Population, int]:
        self.TOTAL_ENTROPY=self.entropy([(self.Data[self.TARGET_VAR] == c).mean() for c in set(self.Data[self.TARGET_VAR])])
        self.Data_Batch= self.get_random_subsets(self.Data)
        population = populate_func(self,popluation_size,genome_length,fitness_func)
        for i in range(generation_limit):
            population = sorted(population, key=lambda genome: fitness_func(self,genome), reverse=True)
            # self.print_stats(population,i,fitness_func)
            next_generation = population[0:2]

            for _ in range(int(len(population) / 2) - 1):
                parents = selection_func(self,population, fitness_func)
                offspring_a, offspring_b = crossover_func(self,parents[0], parents[1])
                offspring_a = mutation_func(self,offspring_a,fitness_func)
                offspring_b = mutation_func(self,offspring_b,fitness_func)
                next_generation += [offspring_a, offspring_b]

            population = next_generation
        population = sorted(population, key=lambda genome: fitness_func(self,genome), reverse=True)
        # print("Best genome: ",end="")
        # print(self.from_genome(population[0],self.Data.columns.to_list()))
        # print()
        return population, i

    def best_feature(self,population:Population,columns:List[str])->None:
        result =[0 for _ in range(len(population[0]['seq']))]
        # print(final_result)
        for gen in population:
            for i,val in enumerate(gen.get('seq')):
                # print(val)
                if val==0:
                    result[i]=result[i]-1
            if(len(result)!=len(columns)):
                columns.remove(self.TARGET_VAR)
        min=0
        best_feature={}
        while len(best_feature)!=len(result):
            for i,val in enumerate(result):
                if val==min:
                    #print(columns[i],":",val)
                    #best_feature.append(columns[i])
                    best_feature[columns[i]]=val
            min=min-1
        return [feature for feature in best_feature.keys() if best_feature[feature]>-2]