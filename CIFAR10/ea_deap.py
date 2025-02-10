"""Mixed-Precision EA
quant_config format = [(w_bit, a_bit) for each layer]
"""
import random
from deap import creator, base, tools, algorithms
import torchvision
import torchvision.transforms as transforms
from models.resnet import *
from test import validate
import torch.backends.cudnn as cudnn


def get_model(quant_config):
    print('=> Building model...')

    model = resnet20_cifar()
    model = nn.DataParallel(model.cuda())

    print("LOADING FULL ACCURACY MODEL")
    ckpt = torch.load("/home/ar7789/LLT/CIFAR10/result/resnet20_w32_a32/best.pth")['state_dict']
    model.load_state_dict(ckpt, strict=False)
    cudnn.benchmark = True

    quant_config_ctr = 0
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            # breakpoint()
            m.w_bits = quant_config[quant_config_ctr][0]
            m.a_bits = quant_config[quant_config_ctr][1]
            quant_config_ctr += 1

    print(quant_config_ctr)
    return model

def get_data():
    print('=> loading cifar10 data...')
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    return test_loader

criterion = nn.CrossEntropyLoss().cuda()
test_loader = get_data()

# EA
NUM_LAYERS = 20 # num of quantization modules in resnet20

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # only one objective to optimize: model accuracy
creator.create("Individual", list, fitness=creator.FitnessMax) # format: list of tupes of (w_bit, a_bit) for each layer

toolbox = base.Toolbox()

def attr_tuple():
    return (random.randint(2, 8), random.randint(2, 8))

toolbox.register("attr_tuple", attr_tuple)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_tuple, n=NUM_LAYERS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_quant_config(individual):
    print(individual)
    model = get_model(individual)
    prec = validate(test_loader, model, criterion),
    print("PREC:", prec)
    return prec

def custom_mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:  # Apply mutation with probability indpb
            mutate_index = random.choice([0, 1])  # Choose whether to mutate w_bit (0) or a_bit (1)
            mutated_tuple = list(individual[i])  # Convert tuple to list (tuples are immutable)
            mutated_tuple[mutate_index] = random.randint(2, 8)  # Mutate the selected index
            individual[i] = tuple(mutated_tuple)  # Convert back to tuple
    return individual,

toolbox.register("evaluate", eval_quant_config)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=5)

NGEN=100
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
top10 = tools.selBest(population, k=10)
print(top10)






# eval_quant_config([(2, 2)] * 20)