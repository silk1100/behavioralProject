from main_experiments import Experiment
import os
import constants
import json
from collections import defaultdict

class ExperimentBuilder:
    def __init__(self, exp_input, output_dir=None):
        if os.path.isdir(exp_input):
            self.input_paths = self._parse_input_path(exp_input)
            self.singleExp = False
        elif os.path.exists(exp_input) and exp_input.endswith('.json'):
            self.input_exp = exp_input
            self.singleExp = True
        else:
            raise ValueError("exp_input should be a path to directory with json files containing experiments or a path"
                             "to a single experiment json file")

        self.main_output = constants.OUTPUT_DIR if output_dir is not None else output_dir
        if self.singleExp:
            # Currently single experiment option is not functional
            if os.name == 'nt':
                self.output_path = os.path.join(self.main_output, self.input_exp.split('\\')[-1].split('.')[0])
            else:
                self.output_path = os.path.join(self.main_output, self.input_exp.split('/')[-1].split('.')[0])
            self._validate_create_output_path(self.output_path)
        else:
            while isinstance(self.input_paths[0], list):
                self.input_paths = sum(self.input_paths, [])
            self.output_paths = self._clone_input_to_output_paths()

        self.experiment_dict = self._read_experiments(self.input_exp) if self.singleExp else \
            self._read_experiments(self.input_paths)
        x=0

    def _clone_input_to_output_paths(self):
        def create_fldr(base, fldr):
            if not os.path.isdir(os.path.join(base, fldr)):
                os.mkdir(os.path.join(base, fldr))
            return os.path.join(base, fldr)

        output_paths = []
        for input_file in self.input_paths:
            if os.name =='nt':
                dirs_2_file = input_file.split('\\')
            else:
                dirs_2_file = input_file.split('/')
            base_dir = self.main_output
            for dir in dirs_2_file[1:]:
                if not dir.endswith('.json'):
                    base_dir = create_fldr(base_dir, dir)
                else:
                    final_dir = create_fldr(base_dir, dir.split('.j')[0])
                    output_paths.append(final_dir)
        return output_paths

    def _parse_input_path(self, path):
        def parse_input_levels(path):
            if (not [x for x in os.listdir(path) if x.endswith('.json')]) and\
                    ([x for x in os.listdir(path) if os.path.isdir(os.path.join(
                        path, x
                    ))]):
                return [parse_input_levels(os.path.join(path, x)) for x in os.listdir(path) if os.path.isdir(os.path.join(
                        path, x
                    ))]
            else:
                return [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.json')]

        return parse_input_levels(path)

    def _validate_create_output_path(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        return True

    def _read_experiments(self, exp_path):
        exp_dict = defaultdict(dict) # {input_path:{"output":str(outputpath), "data": dict(experimentdata)}
        if not self.singleExp:
            for jsonpath, outputpath in zip(exp_path, self.output_paths):
                with open(jsonpath, 'r') as f:
                    data = json.load(f)
                exp_dict[jsonpath]['output'] = outputpath
                exp_dict[jsonpath]['data'] = data

        # Currently it is not functional
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
    #
    # if '-i' in args:
    #     if _validate_input_path(args[args.index('-i')+1]):
    #         input_dir = args[args.index('-i')+1]
    # elif '--input' in args:
    #     if _validate_input_path(args[args.index('--input') + 1]):
    #         input_dir = args[args.index('-i')+1]
    # elif _validate_input_path(args[1]):
    #     input_dir = args[1]
    # else:
    #     raise ValueError("You need to pass an input directory with all experiments within it as the first argument or"
    #                      "-i <dir> or --input <dir>")
    #
    # if '-o' in args:
    #     if _validate_create_output_path(args[args.index('-o')+1]):
    #         output_dir = args[args.index('-o')+1]
    # elif '--output' in args:
    #     if _validate_create_output_path(args[args.index('--output') + 1]):
    #         output_dir = args[args.index('-o')+1]
    # elif _validate_create_output_path(args[2]):
    #     output_dir = args[2]
    # else:
    #     raise ValueError("You need to pass a valid output directory that exists or can be created after"
    #                      "-o <dir> or --output <dir>")

    input_dir = '../experiments/'
    output_dir = "../output/"
    if input_dir.endswith('/') or input_dir.endswith('\\'):
        input_dir = input_dir[:-1]

    exp = ExperimentBuilder(input_dir, output_dir)
    experiments = exp.get_experiment()
    for input_path, experiment_output_dict in experiments.items():
        output_path = experiment_output_dict['output']
        experiment = experiment_output_dict['data']
        experiment['output'] = output_path
        print(f'Running the following Experiment:\n{input_path}')
        e = Experiment(**experiment)
        try:
            e.run()
        except Exception as e:
            print(f"\n\nFollowing error was found during processing {input_path}\n{e}\n\n")
            continue
        print(f'Saving experiment in: {output_path}')
    # exp1 = Experiment(**experiment_1)
    # exp1.run()