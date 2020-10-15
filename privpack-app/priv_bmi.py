from experiments import ExperimentRunner

class BMIExperiment(ExperimentRunner):

    def __init__(self):
        pass

    def run(self, args):
        pass
        ## Figure out the input dimensions: W = (Y, X)
        # Width * Height * Color = 224 * 224 * 3     +   ID 
        # Output: 
        # Width * Height * Color = 224 * 224 * 3

        # Adversary Input:
        # Privatizer out = Width * Height * Color = 224 * 224 * 3
        # Adversary output:
        # Log Likelihood = scalar: likelihood: correct identity.


        # (privacy_size, public_size, hidden_layers_width, release_size) = (5, 5, 20, 5)
        # (epochs, batch_size, lambd, delta, k) = (args.epochs, args.batchsize, args.lambd, args.delta, args.sample)
        # (train_data, test_data) = get_gaussian_data(privacy_size, public_size, print_metrics=True)
        
        # results = {}
        # if len(lambd) == 1 and len(delta) == 1 and len(k) == 1:
        #     runner = GaussianNetworkRunner(privacy_size, public_size, hidden_layers_width, release_size, lambd[0], delta[0])
        #     results = runner.run(train_data, test_data, epochs, batch_size, k[0])
        # else:
        #     runner = GaussianExperiment()
        #     runner.run(train_data, epochs, batch_size, lambd, delta, k)

        # print(json.dumps(results, sort_keys=True, indent=4))
        # if (args.output):
        #     json.dump( results, open( args.output + '.json', 'w' ), indent=4 )
