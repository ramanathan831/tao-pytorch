# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLIP evaluation utilities."""


def create_classifier_templates(templates_file):
    """
    Reads templates from a file and creates lambda functions for each template.

    Args:
        templates_file (str): Path to the file containing template strings.

    Returns:
        tuple: A tuple of lambda functions for each template line in the file.
    """
    templates = []
    with open(templates_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Ignore empty lines
                # Convert the template line into a lambda function
                templates.append(eval(f"lambda c: f'{line}'"))
    return tuple(templates)


def print_templates(templates):
    """
    Prints formatted templates with an example class placeholder.

    Args:
        templates (tuple): A tuple of lambda functions representing the templates.
    """
    for template in templates:
        example_class = "example"
        template_output = template(example_class)
        # Extract the part after 'a photo of' for clarity
        template_str = template_output.replace(example_class, "{c}")
        print(f"Template: {template_str}")


def get_sorted_class_names_and_mapping(classnames):
    """
    Sorts class names and creates a mapping from class name to index.

    Args:
        classnames (list): A list of class names.

    Returns:
        tuple: A tuple of sorted class names and a dictionary mapping each class name to its index.
    """
    sorted_class_names = sorted(classnames)
    class_to_index_mapping = {
        class_name: index
        for index, class_name in enumerate(sorted_class_names)
    }
    return tuple(sorted_class_names), class_to_index_mapping


def create_classnames_mapping(classnames_file):
    """
    Reads class names from a file, sorts them, and creates a class-to-index mapping.

    Args:
        classnames_file (str): Path to the file containing class names.

    Returns:
        tuple: A tuple containing sorted class names and a dictionary mapping each class name to its index.
    """
    with open(classnames_file, 'r') as f:
        classnames = tuple(line.strip() for line in f)
    classnames, mapping = get_sorted_class_names_and_mapping(classnames)
    return classnames, mapping
