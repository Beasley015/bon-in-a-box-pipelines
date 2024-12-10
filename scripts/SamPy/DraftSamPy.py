print("starting...")

import numpy as np
import matplotlib.pyplot as plt
import nlmpy
import random

from sampy.graph.builtin_graph import OrientedHexagonalLattice, SquareGridWithDiag
from sampy.spatial.built_in_graph_intersections import Intersection2DGraphsConvexPolygonalCells
from sampy.addons.ORM_related_addons.ORM_like_agents import ORMLikeAgent
from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity
from sampy.intervention.built_in_interventions import BasicVaccination

# Prep for results storage
output_path = "ORM_outs"

# create the landscape (will remove when real map is created)
map_2d = nlmpy.randomClusterNN(nRow=194, nCol=209, n='8-neighbourhood', p=0.3)
map_2d = nlmpy.classifyArray(map_2d, [0.4, 0.2, 0.2, 0.1, 0.05, 0.05]) # vector of habitat proportions

map_2d = (map_2d+1)*random.randint(5,8) # convert classification IDs to Ks

# Use the map to create a graph attribute for K's
kmap = SquareGridWithDiag(shape=map_2d.shape)
vecsq = (np.sqrt(10), 0.)
kmap.set_coords_from_vector(coord_first_vertex=(-2., -2.), vector=vecsq)
kmap.create_square_vertices((0,1))
kmap.create_attribute_from_2d_array("K", map_2d)

# create hexagonal graph
shape_hex_grid = (100, 100)
vector_hex = (0., np.sqrt(20./np.sqrt(3)))

hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=shape_hex_grid[0], nb_hex_y_axis=shape_hex_grid[1])
hex_grid.set_coords_from_vector((0., 0.), vector_hex)
hex_grid.create_hexagon_vertices()

# Create intersection between the two (needed to migrate K values)
intersection = Intersection2DGraphsConvexPolygonalCells(graph_1=kmap, graph_2=hex_grid)
intersection.compute_intersection_same_nb_vertices()

# Move K values from square map to hex graph 
hex_map_K = intersection.convert_1D_array('g1_to_g2', kmap.df_attributes['K'])
hex_grid.create_vertex_attribute('K', hex_map_K)

# create the population object
agents = ORMLikeAgent(graph=hex_grid)

# add some raccoons [INCREASING STARTING POPULATION CAUSES CRASH IN FOR LOOP]
first_guys = dict()
first_guys['age'] = [random.randint(52, 52*8) for _ in range(hex_grid.number_vertices * 2)] 
first_guys['gender'] = [i % 2 for i in range(hex_grid.number_vertices * 2)]
first_guys['territory'] = [i // 2 for i in range(hex_grid.number_vertices * 2)]
first_guys['position'] = [i // 2 for i in range(hex_grid.number_vertices * 2)]

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

# Create disease object
disease = ContactCustomProbTransitionPermanentImmunity(disease_name='disease', host=agents)

# Probs for transitioning from incubation to infectious
convert_times = np.array(range(1,9))
convert_probs = np.array([0.99, 0.95, 0.75, 0.6, 0.25, 0.05, 0.01, 0])
                                                                  
# Create vaccination object (in the future, separate fixed-wing ORV, ground ORV, and TVR)
vax = BasicVaccination(disease=disease, duration_vaccine=520)
# Absurd duration for essentially permanent immunity

# Create vaccination array
vaxprobs = np.zeros(hex_grid.number_vertices)
vax_indices = [random.randrange(1, hex_grid.number_vertices, 1) for i in range(20)]
vaxprobs[vax_indices] = [random.betavariate(6, 10) for i in range(20)]
# it's random for now, but will make more realistic later

years = 15

# Create blank np array for output storage
outs = np.empty(shape=[0,4])

for i in range(years * 52 + 1):

    agents.tick()
    hex_grid.tick()
    disease.tick()
    vax.update_vaccine_status() # This has to be included for the vax functions to work
    
    # Non-disease mortality
    agents.kill_too_old(52 * 8 - 1)
    agents.natural_death_orm_methodology(weekly_mort, weekly_mort, k_factor_attribute='K')
    agents.kill_children_whose_mother_is_dead(20)
    
    # Run around (some sort of error here)
    agents.mov_around_territory(0.5, condition=agents.df_population['age'] >= 11)
    
    # Disease dynamics
    arr_new_infected = disease.contact_contagion(0.1, return_arr_new_infected=True) # 0.1 is contact rate; change later
    disease.initialize_counters_of_newly_infected(arr_new_infected, convert_times, convert_probs)
    disease.transition_between_states('con', 'death', proba_death=1) 
    disease.transition_between_states('inf', 'con', 
                                      arr_nb_timestep=np.array(convert_times),
                                      arr_prob_nb_timestep=np.array(convert_probs))
    disease.transition_between_states('inf', 'imm', arr_prob_nb_timestep=0.002) # Some never become contagious
    
    # Initialize disease at year 10
    if i == (52*9)+1:
        raw_coords = random.sample(range((min(shape_hex_grid)-1) ** 2), 20)
        sample_coords = [divmod(x, min(shape_hex_grid)-1) for x in raw_coords]

        arr_new_contamination = disease.contaminate_vertices(sample_coords, .5) 

        # Determine how long each agent will be in diseased state
        disease.initialize_counters_of_newly_infected(arr_new_contamination, 
                                              convert_times,  # 1d array of timesteps
                                              convert_probs) # 1d array of prob of staying in incubation at time step i
    
    # Find mate at given time step
    if i % 52 == 9: 
        agents.find_random_mate_on_position(1., position_attribute='territory')

    # Reproduce at a given time step
    if i % 52 == 18: 
        agents.create_offsprings_custom_prob(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), # Vector of litter sizes 
                                             np.array([0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05])) # Prob of each size

    # Vaccination at a given time step
    if i % 52 == 40:
        vax.apply_vaccine_from_array(array_vaccine_level=vaxprobs) #Check if vaxxed inf/con individuals no longer progress

    # Dispersal
    if i % 52 == 45: 
        can_move = agents.df_population['age'] > 20
        agents.dispersion_with_varying_nb_of_steps(np.array([0, 1, 2]), np.array([.9, .09, .01]),
                                                   condition=can_move)
    
    # Start recording at year 10
    if i > (52*9):
        row = [i, sum(agents.count_pop_per_vertex()), 
                sum(agents.count_pop_per_vertex(condition=agents.df_population['inf_disease'])),
                sum(agents.count_pop_per_vertex(condition=agents.df_population['con_disease']))] 
    
        outs = np.vstack([outs, row])

    if i % 52 == 0:
        print(i/52)
        
np.savetxt(output_path + "/outputs.csv", outs, delimiter=',', header="Week, total_pop, n_incubation, n_infectious", comments="")