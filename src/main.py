from main_experiments import Experiment
import os
import constants
import json


class ExperimentBuilder:
    def __init__(self, exp_input, output_dir=None):
        if os.path.isdir(exp_input) and self._validate_input_path(exp_input):
            self.input_path = exp_input
            self.singleExp = False
        elif os.path.exists(exp_input) and exp_input.endswith('.json'):
            self.input_exp = exp_input
            self.singleExp = True
        else:
            raise ValueError("exp_input should be a path to directory with json files containing experiments or a path"
                             "to a single experiment json file")

        self.main_output = constants.OUTPUT_DIR if output_dir is not None else output_dir
        if self.singleExp:
            if os.name == 'nt':
                self.output_path = os.path.join(self.main_output, self.input_exp.split('\\')[-1].split('.')[0])
            else:
                self.output_path = os.path.join(self.main_output, self.input_exp.split('/')[-1].split('.')[0])
            self._validate_create_output_path(self.output_path)
        else:
            if os.name == 'nt':
                self.output_path = os.path.join(self.main_output, self.input_path.split('\\')[-1])
            else:
                self.output_path = os.path.join(self.main_output, self.input_path.split('/')[-1])
            self._validate_create_output_path(self.output_path)
            if os.name == 'nt':
                self.output_subpaths = [os.path.join(self.output_path, x.split('\\')[-1].split('.')[0])
                                        for x in os.listdir(self.input_path) if x.endswith('.json')]
            else:
                self.output_subpaths = [os.path.join(self.output_path, x.split('/')[-1].split('.')[0])
                                        for x in os.listdir(self.input_path) if x.endswith('.json')]

            print(self.output_subpaths)
            _ = [self._validate_create_output_path(path) for path in self.output_subpaths]
        self.experiment_dict = self._read_experiments(self.input_exp) if self.singleExp else \
            self._read_experiments(self.input_path)

    def _validate_input_path(self, path):
        if not os.path.isdir(path):
            return False
        if not [x for x in os.listdir(path) if x.endswith('.json')]:
            return False
        return True

    def _validate_create_output_path(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        return True

    def _read_experiments(self, exp_path):
        exp_dict = {}
        if not self.singleExp:
            exp_paths = [x for x in os.listdir(exp_path) if x.endswith('.json')]
            for exp_json in exp_paths:
                with open(os.path.join(exp_path, exp_json), 'r') as f:
                    exp = json.load(f)
                if os.name == 'nt':
                    exp_dict[exp_json.split('\\')[-1].split('.')[0]] = exp
                    exp_dict[exp_json.split('\\')[-1].split('.')[0]]['output'] =\
                        os.path.join(self.output_path,exp_json.split('\\')[-1].split('.')[0])

                else:
                    exp_dict[exp_json.split('/')[-1].split('.')[0]] = exp
                    exp_dict[exp_json.split('/')[-1].split('.')[0]]['output'] =\
                        os.path.join(self.output_path,exp_json.split('/')[-1].split('.')[0])

        elif isinstance(exp_path, str):
            with open(exp_path, 'r') as f:
                exp = json.load(f)
            if os.name =='nt':
                exp_dict[exp_path.split('\\')[-1].split('.')[0]] = exp
            else:
                exp_dict[exp_path.split('/')[-1].split('.')[0]] = exp

        return exp_dict

    def get_experiment(self):
        return self.experiment_dict

import sys
def main(*args, **kwargs):
    if kwargs['-i'] or kwargs['--input']:
        pass
    elif os.path.isdir(args[0]):
        pass
    else:
        raise ValueError("You need to pass an input directory with all experiments within it as the first argument or"
                         "-i <dir> or --input <dir>")

    if kwargs['-o'] or kwargs['--output']:
        pass
    elif os.path.isdir(args[1]):
        pass
    else:
        raise ValueError("You need to pass an input directory with all experiments within it as the first argument or"
                     "-i <dir> or --input <dir>")


def _validate_input_path(path):
    if os.path.exists(path) and path.endswith('.json'):
        return True
    if not os.path.isdir(path):
        return False
    if not [x for x in os.listdir(path) if x.endswith('.json')]:
        return False
    return True


def _validate_create_output_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return True


if __name__ == "__main__":
    args = sys.argv

    if '-i' in args:
        if _validate_input_path(args[args.index('-i')+1]):
            input_dir = args[args.index('-i')+1]
    elif '--input' in args:
        if _validate_input_path(args[args.index('--input') + 1]):
            input_dir = args[args.index('-i')+1]
    elif _validate_input_path(args[1]):
        input_dir = args[1]
    else:
        raise ValueError("You need to pass an input directory with all experiments within it as the first argument or"
                         "-i <dir> or --input <dir>")

    if '-o' in args:
        if _validate_create_output_path(args[args.index('-o')+1]):
            output_dir = args[args.index('-o')+1]
    elif '--output' in args:
        if _validate_create_output_path(args[args.index('--output') + 1]):
            output_dir = args[args.index('-o')+1]
    elif _validate_create_output_path(args[2]):
        output_dir = args[2]
    else:
        raise ValueError("You need to pass a valid output directory that exists or can be created after"
                         "-o <dir> or --output <dir>")

    if input_dir.endswith('/') or input_dir.endswith('\\'):
        input_dir = input_dir[:-1]

    exp = ExperimentBuilder(input_dir, output_dir)
    experiments = exp.get_experiment()
    for key, experiment in experiments.items():
        print(f'Running the following Experiment:\n{key}')
        e = Experiment(**experiment)
        e.run()
        print(f'Finsihed running experiment: {key}')
    # exp1 = Experiment(**experiment_1)
    # exp1.run()