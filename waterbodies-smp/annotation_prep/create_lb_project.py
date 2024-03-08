#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create Labelbox project and attach ontology

Usage:
    $ python3 create_lb_project.py <project-name>
    

Example:
    $ python3 create_lb_project.py MyProjectExample

Note: Make sure label_ontology.py has your desired labeling ontology.
    See README.md for more info.
"""

import argparse
import labelbox as lb
from _label_ontology import ontology_builder

with open('lb_api_key.txt') as f:
    lines = f.readlines()
    LB_API_KEY = lines[0].rstrip()


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Create Labelbox project and attach labeling ontology',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('project_name',
                   help='Desired name for Labelbox project',
                   type=str)

    return p


def main():
    # Get commandline args
    parser = argparse_init()
    args = parser.parse_args()

    # initiate Labelbox client
    client = lb.Client(api_key=LB_API_KEY)

    # Create project
    project = client.create_project(
            name=args.project_name,
            media_type=lb.MediaType.Image)

    # Create ontology
    ontology = client.create_ontology(
            args.project_name + " Ontology",
            ontology_builder.asdict(),
            media_type=lb.MediaType.Image
            )

    project.setup_editor(ontology)

    print("Labelbox Project ID: ", project.uid)

    return


if __name__ == '__main__':
    main()
