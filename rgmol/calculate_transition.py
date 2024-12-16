#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import objects




def calculate_transition_density(self,grid_points):
    """yes"""
    transition_list = self.properties["transition_list"]
    transition_factor_list = self.properties["transition_factor_list"]


    transition_density_list = []
    for transition in zip(transition_list,transition_factor_list):
        transition_density = 0

        for transition_MO in zip(transition[0],transition[1]):
            MO_OCC = calculate_MO_chosen(self,transition_MO[0],grid_points,delta=3)
            MO_VIRT = calculate_MO_chosen(self,transition_MO[1],grid_points,delta=3)

            transition_density += transition_MO[1] * MO_OCC * MO_VIRT

        transition_density_list.append(transition_density)

    self.properties["transition_density_list"] = transition_density_list
    return transition_density_list





