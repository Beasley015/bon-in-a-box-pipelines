import sys
import subprocess
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sampy-abm'])

import sys, json;
import numpy as np
import random
import geopandas
import csv
import rasterio

from sampy.addons.GIS_interface.geographic_grid import HexGrid
from sampy.addons.ORM_related_addons.ORM_like_agents import ORMLikeAgent
from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity

# Load inputs
inputs = biab_inputs()

# Load raster for grid creation
land = rasterio.open(inputs['rasters'][0])

# Reproject bounding box to Albers Equal Area
bds = land.bounds

bottom_left = (bds[1],bds[0])
bottom_right = (bds[1], bds[2])
top_left = (bds[3],bds[0])
top_right = (bds[3],bds[2])

graph = HexGrid.azimuthal_from_corners(bottom_left, bottom_right, top_left, top_right, cell_area=86.6) 

# Load K geojson
Ks = geopandas.read_file(inputs['K_grid'])

# Add Ks to grid
graph.df_attributes['K'] = Ks['k']

# create the population object
agents = ORMLikeAgent(graph=graph)

# add some raccoons
first_guys = dict()
first_guys['age'] = [random.randint(52, 52*8) for _ in range(int(sum(graph.df_attributes['K'])/2))]
first_guys['gender'] = [i % 2 for i in range(int(sum(graph.df_attributes['K'])/2))]
first_guys['territory'] = [random.randint(0,graph.number_vertices-1) for _ in range(int(sum(graph.df_attributes['K'])/2))]
first_guys['position'] = first_guys['territory'] # start within their territory, but they can move later

agents.add_agents(first_guys)

# Define weekly non-disease mortality
weekly_mort = []
for x in [0.5, 0.3, 0.2, 0.2, 0.2, 0.5, 0.5, 0.6, 1.0]: # these are annual mortalities
    for _ in range(52):
        weekly_mort.append(x)
weekly_mort = np.array(weekly_mort)
weekly_mort = 1 - (1 - weekly_mort) ** (1. / 52.) # convert to weekly mortality

# Create disease object
disease = ContactCustomProbTransitionPermanentImmunity(disease_name='disease', host=agents)

# Probs for transitioning from incubation to infectious
convert_times = np.array(range(1,9))
convert_probs = np.array([0.99, 0.95, 0.75, 0.6, 0.25, 0.05, 0.01, 0])

# Load starting cases
cases = geopandas.read_file(inputs['cases'])
if len(cases.index) != 0:
  dis_indices = [i for i, x in enumerate(cases['case']) if x == 1]

# Define simulation length
years = 6

# Create blank np array for output storage
outs = np.empty(shape=[0,4])

# Creat blank list for maps 
list_fig = []

for i in range(years * 52 + 1):

  agents.tick()
  graph.tick()
  disease.tick()
    
  # Non-disease mortality
  agents.kill_too_old(52 * 8 - 1) 
  agents.natural_death_orm_methodology(weekly_mort, weekly_mort, k_factor_attribute='K')
  agents.kill_children_whose_mother_is_dead(20)

  # Run around 
  agents.mov_around_territory(0.25, condition=agents.df_population['age'] >= 11) 

  # Disease dynamics
  arr_new_infected = disease.contact_contagion(0.001, return_arr_new_infected=True)
  disease.initialize_counters_of_newly_infected(arr_new_infected, convert_times, convert_probs)
  disease.transition_between_states('con', 'death', proba_death=1) 
  disease.transition_between_states('inf', 'con', 
                                arr_nb_timestep=np.array(convert_times),
                                arr_prob_nb_timestep=np.array(convert_probs))
  disease.transition_between_states('inf', 'imm', arr_prob_nb_timestep=0.002) # Some never become contagious

  # Find mate at given time step
  if i % 52 == 9: 
    agents.find_random_mate_on_position(1., position_attribute='territory')

  # Reproduce at a given time step
  if i % 52 == 18: 
    agents.create_offsprings_custom_prob(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), # Vector of litter sizes 
                                          np.array([0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05])) # Prob of each size
    
  # Dispersal
  if i % 52 == 45: 
    can_move = agents.df_population['age'] > 20
    agents.dispersion_with_varying_nb_of_steps(np.array([0, 1, 2]), np.array([.9, .09, .01]),
                                                condition=can_move)
    
  # Initialize disease at year 2
  if i == 53:
    if len(cases.index)!=0:
      arr_new_contamination = disease.contaminate_vertices(list_vertices=[list(graph.dict_cell_id_to_ind.keys())[i] for i in dis_indices],
                                                            level=0.01)
                                
      disease.initialize_counters_of_newly_infected(arr_new_contamination, convert_times, convert_probs)

    else:
      arr_new_contamination = disease.contaminate_vertices(list_vertices=[list(graph.dict_cell_id_to_ind.keys())[i] for i in random.sample(range(0, graph.number_vertices),5)],
                                                             level=0.01)

      # Determine how long each agent will be in diseased state
      disease.initialize_counters_of_newly_infected(arr_new_contamination, 
                                          convert_times,  # 1d array of timesteps
                                          convert_probs) # 1d array of prob of staying in incubation at time step i
    
  row = [i, sum(agents.count_pop_per_vertex()),
        sum(agents.count_pop_per_vertex(condition=agents.df_population['inf_disease'])),
        sum(agents.count_pop_per_vertex(condition=agents.df_population['con_disease']))]
  outs = np.vstack([outs, row])

  list_fig.append(agents.count_pop_per_vertex(position_attribute='territory', condition=agents.df_population['inf_disease'] |
                                                agents.df_population['con_disease']))

# Save outputs
pop_fig_path = output_folder+"/pop_result.csv"
np.savetxt(pop_fig_path, outs, delimiter = ",")  

list_fig_path = output_folder+"/vertices.csv"
with open(list_fig_path, 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerows(list_fig)

biab_output("pop_result", pop_fig_path)
biab_output("cell_values", list_fig_path) 