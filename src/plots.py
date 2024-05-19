
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import os
import torch
import seaborn as sb 
from typing import Tuple, Dict, List, Union
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import wandb
from PIL import Image

from src.utils import find_closest_elem, add_colored_hatches
from src.knowledge_neurons import KnowledgeNeurons
from config import Config


COLORS = {"semantics": '#0173b2', # Colorblind friendly
          "syntax": '#de8f05',
          "knowledge_only": '#029e73',
          "knowledge_shared": '#ece133'}

COLORS_BIS = {"concept": '#0173b2',
              "relation": '#de8f05'}

def plot_KNs_types_all_models(models_analysis, kns_path, plot_error_bars: bool, wandb_flag: bool, p_thresh: float):
    """
        Plot for a threshold (set in config) the different KNs category:
            - sem KNs
            - syn KNs
            - know KNs shared with Autoprompt (resp. Pararel)
            - know KNs only for Pararel (resp. Autoprompt)
            
        For all models (set in config).
    
    """
    
    fig, axs = plt.subplots(2, len(models_analysis.keys()), figsize=(10,5))
    
    max_y = 0
    for k, model_name in enumerate(models_analysis.keys()):
        # Load Res
        pararel_res = models_analysis[model_name]['pararel_sem_syn_know_dist'][0] # [0] because there is only one threshold!
        
        autoprompt_res = models_analysis[model_name]['autoprompt_sem_syn_know_dist'][0]
        
        pararel_res_se = models_analysis[model_name]['pararel_sem_syn_know_dist_se'][0]
        
        autoprompt_res_se = models_analysis[model_name]['autoprompt_sem_syn_know_dist_se'][0]
        
        assert pararel_res[3] == autoprompt_res[3]
        #assert pararel_res[0] == autoprompt_res[0] # This is False I must investigate why!
        print("autoprompt ", autoprompt_res[0])
        print("pararel ", pararel_res[0])
        
        ###
        axs[0,k].bar(1, pararel_res[2], color=COLORS_BIS['concept'])
        axs[0,k].errorbar(1, pararel_res[2], yerr = pararel_res_se[2], color='black')
        
        axs[0,k].bar(2, pararel_res[3], color='none', hatch="///", edgecolor=COLORS_BIS['concept'], lw=4.)
        axs[0,k].errorbar(2, pararel_res[3], yerr = pararel_res_se[3], color='black')
        
        axs[0,k].bar(3, autoprompt_res[2], color='none', edgecolor=COLORS_BIS['concept'], lw=4.)
        axs[0,k].errorbar(3, autoprompt_res[2], yerr = autoprompt_res_se[2], color='black')
        
        axs[1,k].bar(1, pararel_res[1], color=COLORS_BIS['relation'])
        axs[1,k].errorbar(1, pararel_res[1], yerr = pararel_res_se[1], color='black')
        
        axs[1,k].bar(2, pararel_res[0], color='none', hatch="///", edgecolor=COLORS_BIS['relation'], lw=4.)
        axs[1,k].errorbar(2, pararel_res[0], yerr = pararel_res_se[0], color='black')
        
        axs[1,k].bar(3, autoprompt_res[1], color='none', edgecolor=COLORS_BIS['relation'], lw=4.,)
        axs[1,k].errorbar(3, autoprompt_res[1], yerr = autoprompt_res_se[1], color='black')
        
        ### Add legend only once
        if k == 0:
            axs[0,0].bar(-2, 1, color = COLORS_BIS['concept'], label = 'Concept')
            axs[0,0].bar(-3, 1, color = COLORS_BIS['relation'], label = 'Relation')
            axs[0,0].bar(-3.5, 1, color = 'none', label = ' ')
            axs[0,0].bar(-4, 1, color = 'grey', label = 'English')
            axs[0,0].bar(-5, 1, color = 'none', hatch="////", edgecolor='grey', lw=2., label = 'Shared')
            axs[0,0].bar(-6, 1, color = 'none', edgecolor='grey', lw=2., label = 'Autoprompt')
            
        
        ###
        axs[0,k].set_xticks([])
        axs[1,k].set_xticks([])
        axs[0,k].set_xlim((0.5,3.5))
        axs[1,k].set_xlim((0.5,3.5))
        axs[0,k].set_axisbelow(True)
        axs[0,k].yaxis.grid(True)
        axs[1,k].set_axisbelow(True)
        axs[1,k].yaxis.grid(True)
        
        ax0_ymax =int(max([pararel_res[2], pararel_res[2], autoprompt_res[2]]))+2
        fact = 2 if ax0_ymax > 2 else 1
        axs[0,k].set_yticks(np.arange(0, ax0_ymax, fact))
        axs[0,k].set_ylim((0, ax0_ymax))
        ax1_ymax = int(max([pararel_res[1], pararel_res[0], autoprompt_res[1]]))+2
        fact = 2 if ax1_ymax > 2 else 1
        axs[1,k].set_yticks(np.arange(0, ax1_ymax, fact))
        axs[1,k].set_ylim((0, ax1_ymax))
        
        ###
        axs[0,k].set_title(model_name)
        
        # Adjust layout to not overlap
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin to leave space for the legend

        # Create a single legend for all plots on the right side of the figure
        fig.legend(loc='center right', bbox_to_anchor=(0.99, 0.5))
        
        """
        # Pararel
        axs[0].bar(k, pararel_res[0], color = COLORS['semantics'], label = 'sem.')
        if plot_error_bars:
            axs[0].errorbar(k - 1/10, pararel_res[0], yerr = pararel_res_std[0], color =COLORS['semantics'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[0].errorbar(k - 1/10, pararel_res[0], yerr = pararel_res_std[0], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
        
        axs[0].bar(k, pararel_res[1], bottom = pararel_res[0], color = COLORS['syntax'], label = 'syn.')
        if plot_error_bars:
            axs[0].errorbar(k - 1/20, pararel_res[1] + pararel_res[0], yerr = pararel_res_std[1], color =COLORS['syntax'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[0].errorbar(k - 1/20, pararel_res[1] + pararel_res[0], yerr = pararel_res_std[1], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
        
        axs[0].bar(k, pararel_res[3], bottom = pararel_res[0] + pararel_res[1], color = COLORS['knowledge_only'], label = 'know. mono')
        if plot_error_bars:
            axs[0].errorbar(k + 1/20, pararel_res[3] + pararel_res[0] + pararel_res[1], yerr = pararel_res_std[3], color = COLORS['knowledge_only'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[0].errorbar(k + 1/20, pararel_res[3] + pararel_res[0] + pararel_res[1],  yerr = pararel_res_std[3], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
        
        axs[0].bar(k, pararel_res[4], bottom = pararel_res[0] + pararel_res[1] + pararel_res[3], color = COLORS['knowledge_shared'], label = 'know. shared')
        if plot_error_bars:
            axs[0].errorbar(k + 1/10, pararel_res[4] +  pararel_res[0] + pararel_res[1] + pararel_res[3], yerr = pararel_res_std[4], color =COLORS['knowledge_shared'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[0].errorbar(k + 1/10, pararel_res[4] + pararel_res[0] + pararel_res[1] + pararel_res[3], yerr = pararel_res_std[4], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
        
        # Autoprompt
        axs[1].bar(k, autoprompt_res[0], color = COLORS['semantics'], label = 'sem.')
        if plot_error_bars:
            axs[1].errorbar(k - 1/10, autoprompt_res[0], yerr = autoprompt_res_std[0], color =COLORS['semantics'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[1].errorbar(k - 1/10, autoprompt_res[0], yerr = autoprompt_res_std[0], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
            
        axs[1].bar(k, autoprompt_res[1], bottom = autoprompt_res[0], color = COLORS['syntax'], label = 'syn.')
        if plot_error_bars:
            axs[1].errorbar(k - 1/20, autoprompt_res[1] + autoprompt_res[0], yerr = autoprompt_res_std[1], color =COLORS['syntax'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[1].errorbar(k - 1/20, autoprompt_res[1] + autoprompt_res[0], yerr = autoprompt_res_std[1], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
            
        axs[1].bar(k, autoprompt_res[3], bottom = autoprompt_res[0] + autoprompt_res[1], color = COLORS['knowledge_only'], label = 'know. mono')
        if plot_error_bars:
            axs[1].errorbar(k + 1/20, autoprompt_res[3] + autoprompt_res[0] + autoprompt_res[1], yerr = autoprompt_res_std[3], color = COLORS['knowledge_only'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[1].errorbar(k + 1/20, autoprompt_res[3] + autoprompt_res[0] + autoprompt_res[1],  yerr = autoprompt_res_std[3], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
            
        axs[1].bar(k, autoprompt_res[4], bottom = autoprompt_res[0] + autoprompt_res[1] + autoprompt_res[3], color = COLORS['knowledge_shared'], label = 'know. shared')
        if plot_error_bars:
            axs[1].errorbar(k + 1/10, autoprompt_res[4] +  autoprompt_res[0] + autoprompt_res[1] + autoprompt_res[3], yerr = autoprompt_res_std[4], color =COLORS['knowledge_shared'],  elinewidth=2, capsize = 3, capthick = 2, zorder = 10, fmt = 'o')
            axs[1].errorbar(k + 1/10, autoprompt_res[4] + autoprompt_res[0] + autoprompt_res[1] + autoprompt_res[3], yerr = autoprompt_res_std[4], color = 'black',  elinewidth=4, capsize = 4, capthick = 4, zorder = 5, fmt = 'o')
        
        # for ylim
        max_y = max(
                    max(pararel_res[0] + pararel_res[1] + pararel_res[3] + pararel_res[4] + pararel_res_std[3],
                        autoprompt_res[0] + autoprompt_res[1] + autoprompt_res[3] + autoprompt_res[4] + autoprompt_res_std[3]),
                    max_y
                )
       
    axs[0].set_title("Avg. Num. of Sem., Syn. & Know. KNs per UUID \n ParaRel")
    axs[0].set_xticks(np.arange(len(models_analysis)), 
                      labels = list(models_analysis.keys()),)
                      #rotation = 360 - 45)
    axs[0].set_ylim((0,1.1*max_y))
    
    axs[1].set_title("Avg. Num. of Sem., Syn. & Know. KNs per UUID \n Autoprompt")
    axs[1].set_xticks(np.arange(len(models_analysis)), 
                      labels = list(models_analysis.keys()))
    axs[1].set_ylim((0,1.1*max_y))
    plt.legend()
    """
    export_path = os.path.join(
            kns_path,
            f"kns_types_all_models_p_{p_thresh}.png"
        )
    # Save
    plt.savefig(
        export_path,
        dpi = 300
    )
        
    plt.close()
    
    # Log
    if wandb_flag:
        img = Image.open(export_path)
        wandb.log({"KNs Types": [wandb.Image(img)]})
        
        
def plot_sem_syn_know_layer_distribution(models_analysis, threshold: float, kns_path: str, wandb_flag: bool, p_thresh: float):
    
    for model_name in models_analysis.keys():
        threshold = find_closest_elem(models_analysis[model_name]['sem_kns'].keys(), threshold)
        
        fig, axs = plt.subplots(2, 3, figsize=(10,5), sharey='row')
        
        # SEM
        kns = models_analysis[model_name]['sem_kns'][threshold] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        axs[1, 1].bar(np.arange(len(layer_count)), layer_count.values(), color='none', hatch="////", edgecolor=COLORS_BIS['relation'], lw=2.)
        
        # SYN
        kns = models_analysis[model_name]['pararel_syn_kns'][threshold] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        axs[1,0].bar(np.arange(len(layer_count)), layer_count.values(), color = COLORS_BIS['relation'])
        
        kns = models_analysis[model_name]['autoprompt_syn_kns'][threshold] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        axs[1,2].bar(np.arange(len(layer_count)), layer_count.values(), color='none', edgecolor=COLORS_BIS['relation'], lw=2.)
        
        # KNOW
        kns = models_analysis[model_name]['pararel_only_know_kns_set'] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        axs[0,0].bar(np.arange(len(layer_count)), layer_count.values(), color = COLORS_BIS['concept'])
        
        kns = models_analysis[model_name]['shared_know_kns_set'] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        axs[0,1].bar(np.arange(len(layer_count)), layer_count.values(), color='none', hatch="////", edgecolor=COLORS_BIS['concept'], lw=2.)
        
        kns = models_analysis[model_name]['autoprompt_only_know_kns_set'] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        axs[0,2].bar(np.arange(len(layer_count)), layer_count.values(), color='none', edgecolor=COLORS_BIS['concept'], lw=2.)
        
        ###
        axs[0,0].set_xticks([])
        axs[0,1].set_xticks([])
        axs[0,2].set_xticks([])
        
        fig.text(0.5, 0.04, 'Layers', ha='center', va='center', fontsize=12)
        fig.text(0.04, 0.5, 'Neurons Count', ha='center', va='center', rotation='vertical', fontsize=12)
        
        ###
        axs[0,0].bar(-2, 1, color = COLORS_BIS['concept'], label = 'Concept')
        axs[0,0].bar(-3, 1, color = COLORS_BIS['relation'], label = 'Relation')
        axs[0,0].bar(-3.5, 1, color = 'none', label = ' ')
        axs[0,0].bar(-4, 1, color = 'grey', label = 'English')
        axs[0,0].bar(-5, 1, color = 'none', hatch="////", edgecolor='grey', lw=2., label = 'Shared')
        axs[0,0].bar(-6, 1, color = 'none', edgecolor='grey', lw=2., label = 'Autoprompt')
        fig.legend(loc='center right', bbox_to_anchor=(0.99, 0.5))
        
        ###
        axs[0,0].set_xlim((0.5, KnowledgeNeurons.model_layers_num[model_name] + 0.5))
        axs[0,1].set_xlim((0.5, KnowledgeNeurons.model_layers_num[model_name] + 0.5))
        axs[0,2].set_xlim((0.5, KnowledgeNeurons.model_layers_num[model_name] + 0.5))
        axs[1,0].set_xlim((0.5, KnowledgeNeurons.model_layers_num[model_name] + 0.5))
        axs[1,1].set_xlim((0.5, KnowledgeNeurons.model_layers_num[model_name] + 0.5))
        axs[1,2].set_xlim((0.5, KnowledgeNeurons.model_layers_num[model_name] + 0.5))
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.95]) 

        fig.suptitle(model_name, fontsize = 16, y = 0.98)
        
        export_path = os.path.join(
                kns_path,
                model_name,
                f"kns_layer_dist_p_{p_thresh}.png"
                )
        fig.savefig(
                export_path,
                dpi = 300
            )
        

        """
        sem_syn_y_max = 0
        know_y_max = 0
        
        # SEM
        fig_sem = plt.figure()
        kns = models_analysis[model_name]['sem_kns'][threshold] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        plt.bar(np.arange(len(layer_count)), layer_count.values(), color = COLORS['semantics'])
        sem_syn_y_max = max(sem_syn_y_max, max(layer_count.values()))
        
        plt.title('Semantics KNs Layers Distrtibution')
        plt.xlabel('Layer')
        plt.ylabel('Count')
        
        
        
        # SYN
        fig_syn_pararel = plt.figure()
        kns = models_analysis[model_name]['pararel_syn_kns'][threshold] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        plt.bar(np.arange(len(layer_count)), layer_count.values(), color = COLORS['syntax'])
        sem_syn_y_max = max(sem_syn_y_max, max(layer_count.values()))
        
        plt.title('Syntax KNs Layers Distrtibution - ParaRel')
        plt.xlabel('Layer')
        plt.ylabel('Count')
        
        fig_syn_autoprompt = plt.figure()
        kns = models_analysis[model_name]['autoprompt_syn_kns'][threshold] # set
        layer_count = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            layer_count[l] += 1
        plt.bar(np.arange(len(layer_count)), layer_count.values(), color = COLORS['syntax'])
        sem_syn_y_max = max(sem_syn_y_max, max(layer_count.values()))
        
        plt.title('Syntax KNs Layers Distrtibution - Autoprompt')
        plt.xlabel('Layer')
        plt.ylabel('Count')
        
        
        # KNOW
        fig_know_pararel = plt.figure()
        kns = models_analysis[model_name]['pararel_know_kns'][threshold] # set
        kns_dual = models_analysis[model_name]['shared_know_kns'] # set
        layer_count_only = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        layer_count_shared = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            if kn in kns_dual:
                layer_count_shared[l] += 1
            else:
                layer_count_only[l] += 1
        plt.bar(np.arange(len(layer_count_only)), layer_count_only.values(), color = COLORS['knowledge_only'], label = 'mono')
        plt.bar(np.arange(len(layer_count_shared)), layer_count_shared.values(), bottom = list(layer_count_only.values()), color = COLORS['knowledge_shared'], label = 'shared')
        know_y_max = max(know_y_max, 
                         max( np.array(list(layer_count_only.values())) + np.array(list(layer_count_shared.values())) )
                         )
        plt.title('Knowledge KNs Layers Distrtibution - ParaRel')
        plt.xlabel('Layer')
        plt.ylabel('Count')
        plt.legend()
        
        fig_know_autoprompt = plt.figure()
        kns = models_analysis[model_name]['autoprompt_know_kns'][threshold] # set
        layer_count_only = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        layer_count_shared = {k: 0 for k in range(KnowledgeNeurons.model_layers_num[model_name])}
        for kn in kns:
            l, _ = kn
            if kn in kns_dual:
                layer_count_shared[l] += 1
            else:
                layer_count_only[l] += 1
        plt.bar(np.arange(len(layer_count_only)), layer_count_only.values(), color = COLORS['knowledge_only'], label = 'mono')
        plt.bar(np.arange(len(layer_count_shared)), layer_count_shared.values(), bottom = list(layer_count_only.values()), color = COLORS['knowledge_shared'], label = 'shared')
        know_y_max = max(know_y_max, 
                         max( np.array(list(layer_count_only.values())) + np.array(list(layer_count_shared.values())) )
                         )
        plt.title('Knowledge KNs Layers Distrtibution - Autoprompt')
        plt.xlabel('Layer')
        plt.ylabel('Count')
        plt.legend()
        
        
        
        # YLIM
        
        ax_sem = fig_sem.gca()
        ax_sem.set_ylim((0, 1.1*sem_syn_y_max))
        
        ax_syn_pararel = fig_syn_pararel.gca()
        ax_syn_pararel.set_ylim((0, 1.1*sem_syn_y_max))
        
        ax_syn_autoprompt = fig_syn_autoprompt.gca()
        ax_syn_autoprompt.set_ylim((0, 1.1*sem_syn_y_max))
        
        ax_know_pararel = fig_know_pararel.gca()
        ax_know_pararel.set_ylim((0, 1.1*know_y_max))
        
        ax_know_autoprompt = fig_know_autoprompt.gca()
        ax_know_autoprompt.set_ylim((0, 1.1*know_y_max))
        
        
        # SAVING
        
        #Sem
        export_path = os.path.join(
                kns_path,
                model_name,
                'pararel',
                f"sem_kns_layer_dist.png"
                )
        fig_sem.savefig(
                export_path
            )
        if wandb_flag:
            img = Image.open(export_path)
            wandb.log({"Sem. Layer Distribution": [wandb.Image(img)]})
        
        export_path = os.path.join(
                kns_path,
                model_name,
                'autoprompt',
                f"sem_kns_layer_dist.png"
                )
        fig_sem.savefig(
            export_path
            )
        if wandb_flag:
            img = Image.open(export_path)
            wandb.log({"Sem. Layer Distribution (Autoprompt)": [wandb.Image(img)]})
        
        #Syn
        export_path=os.path.join(
                kns_path,
                model_name,
                'pararel',
                f"syn_kns_layer_dist.png"
                )
        fig_syn_pararel.savefig(
            export_path
            )
        if wandb_flag:
            img = Image.open(export_path)
            wandb.log({"Syn. Layer Distribution": [wandb.Image(img)]})
        
        export_path=os.path.join(
                kns_path,
                model_name,
                'autoprompt',
                f"syn_kns_layer_dist.png"
                )
        fig_syn_autoprompt.savefig(
            export_path
            )
        if wandb_flag:
            img = Image.open(export_path)
            wandb.log({"Syn. Layer Distribution (Autoprompt)": [wandb.Image(img)]})
        
        
        #Know
        export_path=os.path.join(
                kns_path,
                model_name,
                'pararel',
                f"know_kns_layer_dist.png"
                )
        fig_know_pararel.savefig(
            export_path
            )
        if wandb_flag:
            img = Image.open(export_path)
            wandb.log({"Know. Layer Distribution": [wandb.Image(img)]})
        
        export_path=os.path.join(
                kns_path,
                model_name,
                'autoprompt',
                f"know_kns_layer_dist.png"
                )
        fig_know_autoprompt.savefig(
            export_path
            )
        if wandb_flag:
            img = Image.open(export_path)
            wandb.log({"Know. Layer Distribution (Autoprompt)": [wandb.Image(img)]})
        
        plt.close('all')
        """



def plot_trex_scores(scores: Union[Tuple[ Dict[str, float], Dict[str, Dict[str, float]] ], Dict[str, Tuple[ Dict[str, float], Dict[str, Dict[str, float]]] ] ],
                     model_name: str,
                     dataset_type: str,
                     wandb_flag: bool = False) -> None:
    
    multilingual = dataset_type[:2] == 'm_'
    if multilingual:
        dataset_type = dataset_type.split('_')[1]
    
    ### P@k ###
    
    if multilingual:
        langs = list(scores.keys())
        ranks = [int(k[2:]) for k in scores[langs[0]][0].keys()]
        for lang in langs:
            plt.plot(ranks, scores[lang][0].values(), marker = '+', label = lang)
        plt.legend()
    else:
        ranks = [int(k[2:]) for k in scores[0].keys()]
        plt.plot(ranks, scores[0].values(), marker = '+')
    
    plt.ylim((0,1))
    plt.xscale('log')
    plt.xticks(ranks, labels=ranks)
    plt.title(f'P@k on TREx - {model_name} - {dataset_type}')
    
    # Save
    results_path = os.path.join('results', 'trex_scores', model_name)
    os.makedirs(results_path, exist_ok=True)
    export_path = os.path.join(
                    results_path,
                    f"p_at_k_{dataset_type}.png"
                )
    plt.savefig(
        export_path
    )
    plt.close()
    
    # Log
    if wandb_flag:
        img = Image.open(export_path)
        wandb.log({"TREx Scores": [wandb.Image(img)]})
    
    ### P@k by rela ###
    
    if multilingual:
        langs = list(scores.keys())
    else:
        langs = [''] # Trick 
        
    for lang in langs:
        fig = plt.figure(figsize=(20,5))
        
        colors = [(0.2, 0, 0), (1, 0, 0)]  # Red, Green, Blue
        # Create a custom colormap
        cmap = LinearSegmentedColormap.from_list('my_gradient', colors)
        
        if multilingual:
            predicate_ids = list(scores[lang][1].keys())
            ranks = [int(k[2:]) for k in scores[lang][0].keys()]
        else:
            predicate_ids = list(scores[1].keys())
            ranks = [int(k[2:]) for k in scores[0].keys()]
            
        
        xs = np.arange(len(predicate_ids))
        norm = LogNorm(vmin=min(ranks), vmax=max(ranks))
        
        for i, predicate_id in enumerate(predicate_ids):
            for k in range(1, len(ranks) + 1):
                if i == 0:
                    label = f'k = {ranks[-k]}'
                else:
                    label = None
                if multilingual:
                    plt.bar([xs[i]], 
                            [scores[lang][1][predicate_id][f'P@{ranks[-k]}']], 
                            width = 2/3, 
                            color=cmap(norm(ranks[-k])), 
                            label = label)
                else:
                    plt.bar([xs[i]], 
                            [scores[1][predicate_id][f'P@{ranks[-k]}']], 
                            width = 2/3, 
                            color=cmap(norm(ranks[-k])), 
                            label = label)
        
        plt.ylim((0,1))
        plt.xticks(xs, labels=predicate_ids, rotation = 360-45)
        plt.legend()
        
        if multilingual:
            plt.title(f'P@k on TREx by Relation - {model_name} - {dataset_type} - {lang}')
        else:
            plt.title(f'P@k on TREx by Relation - {model_name} - {dataset_type}')
        
        
            
        # Save
        results_path = os.path.join('results', 'trex_scores', model_name)
        os.makedirs(results_path, exist_ok=True)
        if multilingual:
            export_path = os.path.join(
                                results_path,
                                f"p_at_k_by_rela_{dataset_type}_{lang}.png"
                            )
        else:
            export_path = os.path.join(
                                results_path,
                                f"p_at_k_by_rela_{dataset_type}.png"
                            )
        plt.savefig(export_path)
        plt.close()
        
        # Log
        if wandb_flag:
            img = Image.open(export_path)
            wandb.log({"TREx Scores by Relation": [wandb.Image(img)]})
    
def plot_kns_surgery(scores: Dict[str, Dict[str, float]],
                     relative_probs: Dict[str, Dict[str, float]], 
                     kns_path: str,
                     kns_match: bool = True,
                     lang: str = '',
                     p_thresh: str = None,
                     config: Config = None) -> None:
    rela_names = list(relative_probs['wo_kns'].keys())
    
    if lang == '':
        title_suffix = ''
        filename_suffix = ''
    else:
        title_suffix = f' (lang = {lang})'
        filename_suffix = f'_{lang}'

    ### PLOT P@k ###
    
    p_at_ks = [f'P@{k}' for k in config.ACCURACY_RANKS]
    ks = config.ACCURACY_RANKS
    
    plt.plot(ks, [scores['vanilla'][k] for k in p_at_ks], linewidth = 3., color = 'black', marker = '+', label = 'vanilla')
    plt.plot(ks, [scores['wo_kns'][k] for k in p_at_ks], linestyle = '--', color = 'grey', marker = '+', label = 'w/o KNs')
    plt.plot(ks, [scores['db_kns'][k] for k in p_at_ks], color = 'grey', marker = '+', label = 'db KNs')
    
    plt.ylim((0,1))
    plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('P@k')
    plt.legend()
    
    if kns_match:
        plt.title(f'P@k on T-REX - Without & Doubling KNs (p = {p_thresh})' + title_suffix)
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery_p_at_k{filename_suffix}_p_{p_thresh}.png"
            )
        )
    else:
        plt.title(f'P@k on T-REX - Without & Doubling KNs (p = {p_thresh}) - UnMatched KNs' + title_suffix)
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery_unmatched_p_at_k{filename_suffix}_p_{p_thresh}.png"
            )
        )
    plt.close()
    
    ### PLOT CCP@k ###
    
    ccp_at_ks = [f'ccp@{k}' for k in config.ACCURACY_RANKS]
    
    plt.plot(ks, [scores['vanilla'][k] for k in ccp_at_ks], linewidth = 3., color = 'black', marker = '+', label = 'vanilla')
    plt.plot(ks, [scores['wo_kns'][k] for k in ccp_at_ks], linestyle = '--', color = 'grey', marker = '+', label = 'w/o KNs')
    plt.plot(ks, [scores['db_kns'][k] for k in ccp_at_ks], color = 'grey', marker = '+', label = 'db KNs')
    
    plt.ylim((0,1))
    plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('CCP@k')
    plt.legend()
    
    if kns_match:
        plt.title(f'CCP@k on T-REX - Without & Doubling KNs (p = {p_thresh})' + title_suffix)
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery_ccp_at_k{filename_suffix}_p_{p_thresh}.png"
            )
        )
    else:
        plt.title(f'CCP@k on T-REX - Without & Doubling KNs (p = {p_thresh})- UnMatched KNs' + title_suffix)
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery_unmatched_ccp_at_k{filename_suffix}_p_{p_thresh}.png"
            )
        )
    plt.close()

    
    ### PLOT RELATIVE PROBS CHANGE ###
    
    # Params
    n_k = 1
    n_relas = len(rela_names)
    n_bars_per_rela = n_k * 2
    colors = ['cornflowerblue', 'navy']

    fig = plt.figure(figsize = (0.5*n_relas, 5))
    for i in range(n_relas):
        width = 1/(n_bars_per_rela + 1)
        plt.bar(i + 1*width + width/2, relative_probs['wo_kns'][rela_names[i]], width = width, color=colors[0])
        plt.bar(i + 2*width + width/2, relative_probs['db_kns'][rela_names[i]], width = width, color=colors[1])

    plt.hlines(0, xmin=0, xmax=n_relas, color='black')
    plt.xlim((0,n_relas))
    plt.ylim((-5,5))
    plt.xticks(np.arange(n_relas) + 0.5, rela_names)

    # Legend
    wo_patch = mpatches.Patch(facecolor=colors[0], label='w/o KNs')
    db_patch = mpatches.Patch(facecolor=colors[1], label='db KNs')
    plt.legend(handles=[wo_patch, db_patch])
    if kns_match:
        plt.title(f'Relative Probs - Without KNs, Doubling KNs  (p = {p_thresh})' + title_suffix)
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery_relative_probs{filename_suffix}_p_{p_thresh}.png"
            )
        )
    else:
        plt.title(f'Relative Probs - Without KNs, Doubling KNs  (p = {p_thresh}) - UnMatched KNs' + title_suffix)
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery_unmatched_relative_probs{filename_suffix}_p_{p_thresh}.png"
            )
        )
    plt.close()
    

def plot_multilingual_analysis(res, **kwargs):
    
    layers_count, sem_layers_count, syn_layers_count, shared_know_layers_count, only_know_layers_count, heatmap, sem_heatmap, syn_heatmap, shared_know_heatmap, only_know_heatmap = res
    MIN_ALPHA = 0.2
    MAX_ALPHA = 1.
    ALPHA_OFFSET = (MAX_ALPHA - MIN_ALPHA)/(len(layers_count)-1)
    
    ### LAYER DISTRIBUTION ###
    
    # KNs
    fig = plt.figure()
    bottom = None
    alpha = MIN_ALPHA
    for l in layers_count.keys():
        if bottom is not None:
            bottom += np.array(list(layers_count[l-1].values()))
        else:
            bottom = np.zeros(len(layers_count[l]))
        plt.bar(list(layers_count[l].keys()), 
                list(layers_count[l].values()), 
                bottom = bottom,
                color = 'gray',
                alpha = alpha,
                label = f'{l}')
        alpha += ALPHA_OFFSET
    know_y_max = max( bottom + np.array(list(layers_count[l].values())))
    plt.xlabel('Layer')
    plt.ylabel('KNs Count')
    
    plt.legend()
    plt.title(f"KNs Multilinguality  (p = {kwargs['p_thresh']}) - {kwargs['dataset_type']}")
 
    
    sem_syn_y_max = 0
    # SEM
    sem_fig = plt.figure()
    bottom = None
    alpha = MIN_ALPHA
    for l in sem_layers_count.keys():
        if bottom is not None:
            bottom += np.array(list(sem_layers_count[l-1].values()))
        else:
            bottom = np.zeros(len(sem_layers_count[l]))
        plt.bar(list(sem_layers_count[l].keys()), 
                list(sem_layers_count[l].values()), 
                bottom = bottom,
                color = 'none',
                hatch = "///",
                edgecolor=COLORS_BIS['relation'],
                alpha = alpha,
                lw=2.,
                label = f'{l}')
        alpha += ALPHA_OFFSET
    
    sem_syn_y_max = max(
                        sem_syn_y_max,
                        max( bottom + np.array(list(sem_layers_count[l].values())) )
                    )
    
    plt.xlabel('Layer')
    plt.ylabel('KNs Count')
    
    plt.legend()
    plt.title(f"Semantics KNs Multilinguality  (p = {kwargs['p_thresh']}) - {kwargs['dataset_type']}")
    
    # SYN
    syn_fig = plt.figure()
    bottom = None
    alpha = MIN_ALPHA
    for l in syn_layers_count.keys():
        if bottom is not None:
            bottom += np.array(list(syn_layers_count[l-1].values()))
        else:
            bottom = np.zeros(len(syn_layers_count[l]))
        plt.bar(list(syn_layers_count[l].keys()), 
                list(syn_layers_count[l].values()), 
                bottom = bottom,
                color = COLORS_BIS['relation'],
                alpha = alpha,
                label = f'{l}')
        alpha += ALPHA_OFFSET
        
    sem_syn_y_max = max(
                        sem_syn_y_max,
                        max( bottom + np.array(list(syn_layers_count[l].values())) )
                    )
    
    plt.xlabel('Layer')
    plt.ylabel('KNs Count')
    
    plt.legend()
    plt.title(f"Syntax KNs Multilinguality  (p = {kwargs['p_thresh']}) - {kwargs['dataset_type']}")
 
    
    
    # KNOW
    only_know_fig = plt.figure()
    bottom = None
    alpha = MIN_ALPHA
    for l in only_know_layers_count.keys():
        if bottom is not None:
            bottom += np.array(list(only_know_layers_count[l-1].values()))
        else:
            bottom = np.zeros(len(only_know_layers_count[l]))
        plt.bar(list(only_know_layers_count[l].keys()), 
                list(only_know_layers_count[l].values()), 
                bottom = bottom,
                color = COLORS_BIS['concept'],
                alpha = alpha,
                label = f'{l}')
        alpha += ALPHA_OFFSET
        
    plt.xlabel('Layer')
    plt.ylabel('KNs Count')
    
    plt.legend()
    plt.title(f"Concept (English only) KNs Multilinguality  (p = {kwargs['p_thresh']}) - {kwargs['dataset_type']}")
    
    shared_know_fig = plt.figure()
    bottom = None
    alpha = MIN_ALPHA
    for l in shared_know_layers_count.keys():
        if bottom is not None:
            bottom += np.array(list(shared_know_layers_count[l-1].values()))
        else:
            bottom = np.zeros(len(shared_know_layers_count[l]))
        plt.bar(list(shared_know_layers_count[l].keys()), 
                list(shared_know_layers_count[l].values()), 
                bottom = bottom,
                color = 'none',
                hatch = '///',
                edgecolor = COLORS_BIS['concept'],
                lw = 2.,
                alpha = alpha,
                label = f'{l}')
        alpha += ALPHA_OFFSET
        
    plt.xlabel('Layer')
    plt.ylabel('KNs Count')
    
    plt.legend()
    plt.title(f"Concept (Shared with AutoPrompt) KNs Multilinguality  (p = {kwargs['p_thresh']}) - {kwargs['dataset_type']}")
 
    # YLIM
    
    ax_sem = sem_fig.gca()
    ax_sem.set_ylim((0, 1.1*sem_syn_y_max))
    
    ax_syn = syn_fig.gca()
    ax_syn.set_ylim((0, 1.1*sem_syn_y_max))    
    
    ax = fig.gca()
    ax.set_ylim((0, 1.1*know_y_max))
    
    ax_only_know = only_know_fig.gca()
    ax_only_know.set_ylim((0, 1.1*know_y_max)) 
    
    ax_shared_know = shared_know_fig.gca()
    ax_shared_know.set_ylim((0, 1.1*know_y_max)) 
    
    # SAVING
    
    fig.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"kns_multilinguality_layer_dist_p_{kwargs['p_thresh']}.png"
            )
        )
    
    sem_fig.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"sem_kns_multilinguality_layer_dist_p_{kwargs['p_thresh']}.png"
            )
        )
    
    syn_fig.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"syn_kns_multilinguality_layer_dist_p_{kwargs['p_thresh']}.png"
            )
        )
    
    only_know_fig.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"only_know_kns_multilinguality_layer_dist_p_{kwargs['p_thresh']}.png"
            )
        )
    
    shared_know_fig.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"shared_know_kns_multilinguality_layer_dist_p_{kwargs['p_thresh']}.png"
            )
        )
    
    plt.close('all')
    
    ### HEATMAPS ###
    
    # Mask the upper triangular part of the array
    mask = np.triu(np.ones_like(heatmap, dtype=bool))
    np.fill_diagonal(mask, False)

    dataplot = sb.heatmap(heatmap, 
                          cmap="Greys", 
                          annot=True,
                          xticklabels= kwargs['config'].LANGS, 
                          yticklabels= kwargs['config'].LANGS,
                          fmt='.0f',
                          mask=mask) 

    plt.title(f'Shared KNs across Languages  (p = {kwargs["p_thresh"]})')
    plt.xlabel('Languages')
    plt.ylabel('Languages')
    
    plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"heatmap_kns_multilinguality_p_{kwargs['p_thresh']}.png"
            )
        )
    plt.close()
    
    # SEM
    
    fig, ax = plt.subplots()
    cmap = sb.color_palette("Oranges", as_cmap=True)
    
    mask = np.triu(np.ones_like(sem_heatmap, dtype=bool))
    np.fill_diagonal(mask, False)

    # Create the heatmap
    sb.heatmap(sem_heatmap, 
               cmap=ListedColormap(['white']), 
               annot=True, 
               cbar = False, 
               xticklabels=kwargs['config'].LANGS, 
               yticklabels=kwargs['config'].LANGS,
               fmt='.0f', 
               ax=ax, 
               mask = mask)

    # Generate colors for hatches based on cell value
    colors = cmap(sem_heatmap/sem_heatmap.max())

    # Add colored hatches
    add_colored_hatches(ax, sem_heatmap, colors)
    
    # Customized Color Bar
    norm = Normalize(vmin=np.min(sem_heatmap), vmax=np.max(sem_heatmap))
    sm = ScalarMappable(cmap="Oranges", norm=norm)
    sm.set_array([])  # You can set an array of values to match the range you are using

    # Create color bar for this ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    # Adjust the color bar container to remove the black edge
    cbar.outline.set_edgecolor(None)

    ax.set_title(f'Shared Semantics KNs across Languages  (p = {kwargs["p_thresh"]})')
    
    plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"sem_heatmap_kns_multilinguality_p_{kwargs['p_thresh']}.png"
            )
        )
    plt.close()
    
    # SYN
    mask = np.triu(np.ones_like(syn_heatmap, dtype=bool))
    np.fill_diagonal(mask, False)

    dataplot = sb.heatmap(syn_heatmap, 
                          cmap="Oranges", 
                          annot=True,
                          xticklabels= kwargs['config'].LANGS, 
                          yticklabels= kwargs['config'].LANGS,
                          fmt='.0f',
                          mask=mask) 

    plt.title(f'Shared Syntax KNs across Languages  (p = {kwargs["p_thresh"]})')
    plt.xlabel('Languages')
    plt.ylabel('Languages')
    
    plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"syn_heatmap_kns_multilinguality_p_{kwargs['p_thresh']}.png"
            )
        )
    plt.close()
    
    # KNOW
    mask = np.triu(np.ones_like(only_know_heatmap, dtype=bool))
    np.fill_diagonal(mask, False)

    dataplot = sb.heatmap(only_know_heatmap, 
                          cmap="Blues", 
                          annot=True,
                          xticklabels= kwargs['config'].LANGS, 
                          yticklabels= kwargs['config'].LANGS,
                          fmt='.0f',
                          mask=mask) 

    plt.title(f'Shared Knowledge (English only) KNs \nacross Languages  (p = {kwargs["p_thresh"]})')
    plt.xlabel('Languages')
    plt.ylabel('Languages')
    
    plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"only_know_heatmap_kns_multilinguality_p_{kwargs['p_thresh']}.png"
            )
        )
    plt.close()
    
    
    fig, ax = plt.subplots()
    cmap = sb.color_palette("Blues", as_cmap=True)
    
    mask = np.triu(np.ones_like(shared_know_heatmap, dtype=bool))
    np.fill_diagonal(mask, False)

    # Create the heatmap
    sb.heatmap(shared_know_heatmap, 
               cmap=ListedColormap(['white']), 
               annot=True, 
               cbar = False, 
               xticklabels=kwargs['config'].LANGS, 
               yticklabels=kwargs['config'].LANGS,
               fmt='.0f', 
               ax=ax, 
               mask = mask)

    # Generate colors for hatches based on cell value
    colors = cmap(shared_know_heatmap/shared_know_heatmap.max())

    # Add colored hatches
    add_colored_hatches(ax, shared_know_heatmap, colors)
    
    # Customized Color Bar
    norm = Normalize(vmin=np.min(shared_know_heatmap), vmax=np.max(shared_know_heatmap))
    sm = ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])  # You can set an array of values to match the range you are using

    # Create color bar for this ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    # Adjust the color bar container to remove the black edge
    cbar.outline.set_edgecolor(None)
    
    ax.set_title(f'Shared Knowledge (Shared with Autoprompt) KNs \n across Languages  (p = {kwargs["p_thresh"]})')
    
    plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset_type'],
                f"shared_know_heatmap_kns_multilinguality_p_{kwargs['p_thresh']}.png"
            )
        )
    plt.close()
    
def plot_kns_exps(
            scores: Dict[str, Dict[str, float]],
            kns_path: str,
            config: Config,
            **kwargs
            ) -> None:

    if 1 in scores.keys():
        ### PLOT EXP 1 ###
        
        # P@k #
        fig = plt.figure(figsize=(7,10))
        
        p_at_ks = [f'P@{k}' for k in config.ACCURACY_RANKS]
        ks = config.ACCURACY_RANKS
        
        marker = "P"
        markersize = 10.
        
        plt.hlines(y = 0., xmin=min(ks), xmax=max(ks), color='grey', linewidth=2.)
        
        plt.plot(ks, 
                 [scores[1]['sem_wo_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['sem_db_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[1]['syn_wo_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['syn_db_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[1]['only_know_wo_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['only_know_db_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[1]['shared_know_wo_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linestyle='--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['shared_know_db_kns'][k] - scores[1]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'],
                 linestyle = '--', 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        
        
        
        """
        plt.plot(np.arange(4), [scores[1]['vanilla'][k] for k in p_at_ks], marker = '+', linewidth = 3., color='black', label = 'vanilla')
        plt.plot(np.arange(4), [scores[1]['sem_wo_kns'][k] for k in p_at_ks], color = COLORS['semantics'], linestyle = '--', marker = '+', label = ' w/o Sem KNs')
        plt.plot(np.arange(4), [scores[1]['sem_db_kns'][k] for k in p_at_ks], color = COLORS['semantics'], marker = '+', label = 'db Sem KNs')
        plt.plot(np.arange(4), [scores[1]['syn_wo_kns'][k] for k in p_at_ks], color = COLORS['syntax'], linestyle = '--', marker = '+', label = ' w/o Syn KNs')
        plt.plot(np.arange(4), [scores[1]['syn_db_kns'][k] for k in p_at_ks], color = COLORS['syntax'], marker = '+', label = 'db Syn KNs')
        plt.plot(np.arange(4), [scores[1]['know_wo_kns'][k] for k in p_at_ks], color = COLORS['knowledge_only'], linestyle = '--', marker = '+', label = ' w/o Know KNs')
        plt.plot(np.arange(4), [scores[1]['know_db_kns'][k] for k in p_at_ks], color = COLORS['knowledge_only'], marker = '+', label = 'db Know KNs')
        
        plt.ylim((0,1))
        """
        
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['concept'], label = 'Concept')
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['relation'], label = 'Relation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'none', label = ' ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 2., label = 'English')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linestyle='--', linewidth = 2., label = 'Shared')
        plt.plot([-5,-4], [0.0, 0.0], marker = '+', color = 'none', label = '  ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 3., label = '2xActivation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 1., label = '0xActivation')
        
        
        
        plt.xlim((min(ks), max(ks)))
        plt.xticks(ks, 
                   labels = config.ACCURACY_RANKS)
        plt.xscale('log')
        plt.xlabel('k')
        plt.ylabel(r'$\Delta$P@k')
        plt.title(f'P@k on T-REX - Without & Doubling KNs\n p = {kwargs["p_thresh"]} - {kwargs["dataset_name"]} - {scores[1]["kns_mode"]} - {np.round(scores[1]["threshold"],2)}')
        
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right margin to leave space for the legend

        # Create a single legend for all plots on the right side of the figure
        fig.legend(loc='center right', bbox_to_anchor=(0.99, 0.5))
        
        # Save
        plt.savefig(
            os.path.join(
                kns_path,
                kwargs["dataset_name"],
                f"kns_exp_1_p_at_k_thresh_{int(scores[1]['threshold']*100)}_{scores[1]['kns_mode']}_p_{kwargs['p_thresh']}.png"
            )
        )
        plt.close()
        
        """
        # Bar plot
        plt.bar([1,3,5], [scores[1]['sem_wo_kns']['P@1'],
                          scores[1]['syn_wo_kns']['P@1'],
                          scores[1]['know_wo_kns']['P@1']],
                color = [COLORS['semantics'], COLORS['syntax'], COLORS['knowledge_only']],
                hatch = '///')
        plt.bar([2,4,6], [scores[1]['sem_db_kns']['P@1'],
                          scores[1]['syn_db_kns']['P@1'],
                          scores[1]['know_db_kns']['P@1']],
                color = [COLORS['semantics'], COLORS['syntax'], COLORS['knowledge_only']])
        plt.hlines(y = scores[1]['vanilla']['P@1'], xmin = 0, xmax = 7, color = 'black', label = 'vanilla')
        
        plt.bar([-5], [1], color = ['white'], edgecolor='black', hatch = '///', label = 'w/o')
        plt.bar([-6], [1], color = ['white'], edgecolor='black', label = 'double')
        
        plt.xlim((0, 7))
        plt.ylim((0,1))
        plt.xticks([1.5,3.5,5.5], labels=['sem.', 'syn.', 'know'])
        plt.legend()
        plt.grid(True)
        plt.title(f'p = {kwargs["p_thresh"]}')

        plt.savefig(
            os.path.join(
                kns_path,
                kwargs["dataset_name"],
                f"kns_exp_1_p_at_1_thresh_{int(scores[1]['threshold']*100)}_{scores[1]['kns_mode']}_bar_plot_p_{kwargs['p_thresh']}.png"
            )
        )
        plt.close()
        """
        # CCP@k #
        
        ccp_at_ks = [f'ccp@{k}' for k in config.ACCURACY_RANKS]
        
                # P@k #
        fig = plt.figure(figsize=(7,10))
        
        ks = config.ACCURACY_RANKS
        
        marker = "P"
        markersize = 10.
        
        plt.hlines(y = 0., xmin=min(ks), xmax=max(ks), color='grey', linewidth=2.)
        
        plt.plot(ks, 
                 [scores[1]['sem_wo_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['sem_db_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[1]['syn_wo_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['syn_db_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[1]['only_know_wo_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['only_know_db_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[1]['shared_know_wo_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linestyle='--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[1]['shared_know_db_kns'][k] - scores[1]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['concept'],
                 linestyle = '--', 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['concept'], label = 'Concept')
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['relation'], label = 'Relation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'none', label = ' ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 2., label = 'English')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linestyle='--', linewidth = 2., label = 'Shared')
        plt.plot([-5,-4], [0.0, 0.0], marker = '+', color = 'none', label = '  ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 3., label = '2xActivation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 1., label = '0xActivation')
        
        
        
        plt.xlim((min(ks), max(ks)))
        plt.xticks(ks, 
                   labels = config.ACCURACY_RANKS)
        plt.xscale('log')
        plt.xlabel('k')
        plt.ylabel(r'$\Delta$CCP@k')
        plt.title(f'CCP@k on T-REX - Without & Doubling KNs\n p = {kwargs["p_thresh"]} - {kwargs["dataset_name"]} - {scores[1]["kns_mode"]} - {np.round(scores[1]["threshold"],2)}')
        
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right margin to leave space for the legend

        # Create a single legend for all plots on the right side of the figure
        fig.legend(loc='center right', bbox_to_anchor=(0.99, 0.5))
        
        """
        plt.plot(np.arange(4), [scores[1]['vanilla'][k] for k in ccp_at_ks], marker = '+', linewidth = 3., color='black', label = 'vanilla')
        plt.plot(np.arange(4), [scores[1]['sem_wo_kns'][k] for k in ccp_at_ks], color = COLORS['semantics'], linestyle = '--', marker = '+', label = ' w/o Sem KNs')
        plt.plot(np.arange(4), [scores[1]['sem_db_kns'][k] for k in ccp_at_ks], color = COLORS['semantics'], marker = '+', label = 'db Sem KNs')
        plt.plot(np.arange(4), [scores[1]['syn_wo_kns'][k] for k in ccp_at_ks], color = COLORS['syntax'], linestyle = '--', marker = '+', label = ' w/o Syn KNs')
        plt.plot(np.arange(4), [scores[1]['syn_db_kns'][k] for k in ccp_at_ks], color = COLORS['syntax'], marker = '+', label = 'db Syn KNs')
        plt.plot(np.arange(4), [scores[1]['know_wo_kns'][k] for k in ccp_at_ks], color = COLORS['knowledge_only'], linestyle = '--', marker = '+', label = ' w/o Know KNs')
        plt.plot(np.arange(4), [scores[1]['know_db_kns'][k] for k in ccp_at_ks], color = COLORS['knowledge_only'], marker = '+', label = 'db Know KNs')
        
        plt.ylim((0,1))
        plt.xticks(np.arange(4), labels = config.ACCURACY_RANKS)
        plt.xlabel('k')
        plt.ylabel('CCP@k')
        plt.legend()
        plt.title(f'CCP@k on T-REX - Without & Doubling KNs Semantics & Syntax KNs\n p = {kwargs["p_thresh"]} -{kwargs["dataset_name"]} - {scores[1]["kns_mode"]} - {np.round(scores[1]["threshold"],2)}')
        """
            
        # Save
        plt.savefig(
            os.path.join(
                kns_path,
                kwargs['dataset_name'],
                f"kns_exp_1_ccp_thresh_{int(scores[1]['threshold']*100)}_{scores[1]['kns_mode']}_p_{kwargs['p_thresh']}.png"
            )
        )
        plt.close()
        
        """
        # Bar plot
        plt.bar([1,3,5], [scores[1]['sem_wo_kns']['ccp@1'],
                          scores[1]['syn_wo_kns']['ccp@1'],
                          scores[1]['know_wo_kns']['ccp@1']],
                color = [COLORS['semantics'], COLORS['syntax'], COLORS['knowledge_only']],
                hatch = '///')
        plt.bar([2,4,6], [scores[1]['sem_db_kns']['ccp@1'],
                          scores[1]['syn_db_kns']['ccp@1'],
                          scores[1]['know_db_kns']['ccp@1']],
                color = [COLORS['semantics'], COLORS['syntax'], COLORS['knowledge_only']])
        plt.hlines(y = scores[1]['vanilla']['ccp@1'], xmin = 0, xmax = 7, color = 'black', label = 'vanilla')
        
        plt.bar([-5], [1], color = ['white'], edgecolor='black', hatch = '///', label = 'w/o')
        plt.bar([-6], [1], color = ['white'], edgecolor='black', label = 'double')
        
        plt.xlim((0, 7))
        plt.ylim((0,1))
        plt.xticks([1.5,3.5,5.5], labels=['sem.', 'syn.', 'know'])
        plt.legend()
        plt.grid(True)
        plt.title(f'p = {kwargs["p_thresh"]}')
        
        plt.savefig(
            os.path.join(
                kns_path,
                kwargs["dataset_name"],
                f"kns_exp_1_ccp_at_1_thresh_{int(scores[1]['threshold']*100)}_{scores[1]['kns_mode']}_bar_plot_p_{kwargs['p_thresh']}.png"
            )
        )
        plt.close()
        """
        
    if 2 in scores.keys():
        ### PLOT EXP 2 ###
        
        # P@k #
        fig = plt.figure(figsize=(7,10))
        
        p_at_ks = [f'P@{k}' for k in config.ACCURACY_RANKS]
        ks = config.ACCURACY_RANKS
        
        marker = "P"
        markersize = 10.
        
        plt.hlines(y = 0., xmin=min(ks), xmax=max(ks), color='grey', linewidth=2.)
        
        plt.plot(ks, 
                 [scores[2]['sem_wo_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['sem_db_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[2]['syn_wo_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['syn_db_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[2]['only_know_wo_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['only_know_db_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[2]['shared_know_wo_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linestyle='--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['shared_know_db_kns'][k] - scores[2]['vanilla'][k]for k in p_at_ks], 
                 color = COLORS_BIS['concept'],
                 linestyle = '--', 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['concept'], label = 'Concept')
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['relation'], label = 'Relation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'none', label = ' ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 2., label = 'English')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linestyle='--', linewidth = 2., label = 'Shared')
        plt.plot([-5,-4], [0.0, 0.0], marker = '+', color = 'none', label = '  ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 3., label = '2xActivation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 1., label = '0xActivation')
        
        
        
        plt.xlim((min(ks), max(ks)))
        plt.xticks(ks, 
                   labels = config.ACCURACY_RANKS)
        plt.xscale('log')
        plt.xlabel('k')
        plt.ylabel(r'$\Delta$P@k')
        plt.title(f'P@k on T-REX - Without & Doubling KNs\n On Trivial Prompt "X [MASK] ." \n p = {kwargs["p_thresh"]} - {kwargs["dataset_name"]} - {scores[1]["kns_mode"]} - {np.round(scores[1]["threshold"],2)}')
        
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right margin to leave space for the legend

        # Create a single legend for all plots on the right side of the figure
        fig.legend(loc='center right', bbox_to_anchor=(0.99, 0.5))
        
        
        """
        plt.plot(np.arange(4), [scores[2]['vanilla'][k] for k in p_at_ks], marker = '+', linewidth = 3., color='black', label = 'vanilla')
        plt.plot(np.arange(4), [scores[2]['sem_wo_kns'][k] for k in p_at_ks], color = 'blue', linestyle = '--', marker = '+', label = ' w/o Sem KNs')
        plt.plot(np.arange(4), [scores[2]['sem_db_kns'][k] for k in p_at_ks], color = 'blue', marker = '+', label = 'db Sem KNs')
        plt.plot(np.arange(4), [scores[2]['know_wo_kns'][k] for k in p_at_ks], color = 'green', linestyle = '--', marker = '+', label = ' w/o Know KNs')
        plt.plot(np.arange(4), [scores[2]['know_db_kns'][k] for k in p_at_ks], color = 'green', marker = '+', label = 'db Know KNs')
        plt.plot(np.arange(4), [scores[2]['syn_wo_kns'][k] for k in p_at_ks], color = 'red', linestyle = '--', marker = '+', label = ' w/o Syn KNs')
        plt.plot(np.arange(4), [scores[2]['syn_db_kns'][k] for k in p_at_ks], color = 'red', marker = '+', label = 'db Syn KNs')
        
        plt.ylim((0,0.3))
        plt.xticks(np.arange(4), 
                labels = config.ACCURACY_RANKS)
        plt.xlabel('k')
        plt.ylabel('P@k')
        plt.legend()
        plt.title(f'P@k on T-REX - Without & Doubling KNs Semantics & Knowledge KNs\n On Trivial Prompt "X [MASK] ."\n p = {kwargs["p_thresh"]} - {kwargs["dataset_name"]} - {scores[2]["kns_mode"]} - {np.round(scores[2]["threshold"],2)}')
        #plt.gcf().subplots_adjust(top=0.15)
        """
        
            
        # Save
        plt.savefig(
            os.path.join(
                kns_path,
                kwargs['dataset_name'],
                f"kns_exp_2_p_at_k_thresh_{int(scores[2]['threshold']*100)}_{scores[2]['kns_mode']}_{kwargs['dataset_name']}_db_fact_{scores[2]['db_fact']}_p_{kwargs['p_thresh']}.png"
            )
        )
        plt.close()
        
        # CCP@k #
        
        ccp_at_ks = [f'ccp@{k}' for k in config.ACCURACY_RANKS]
        
                
                # P@k #
        fig = plt.figure(figsize=(7,10))
        
        ks = config.ACCURACY_RANKS
        
        marker = "P"
        markersize = 10.
        
        plt.hlines(y = 0., xmin=min(ks), xmax=max(ks), color='grey', linewidth=2.)
        
        plt.plot(ks, 
                 [scores[2]['sem_wo_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['sem_db_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linestyle = '--',
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[2]['syn_wo_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['syn_db_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['relation'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[2]['only_know_wo_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['only_know_db_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot(ks, 
                 [scores[2]['shared_know_wo_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color = COLORS_BIS['concept'], 
                 linestyle='--',
                 linewidth = 1., 
                 marker = marker,
                 markersize=markersize)
        plt.plot(ks, 
                 [scores[2]['shared_know_db_kns'][k] - scores[2]['vanilla'][k]for k in ccp_at_ks], 
                 color =COLORS_BIS['concept'],
                 linestyle = '--', 
                 linewidth = 3., 
                 marker = marker,
                 markersize=markersize)
        
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['concept'], label = 'Concept')
        plt.plot([-5,-4], [0.0, 0.0], color = COLORS_BIS['relation'], label = 'Relation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'none', label = ' ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 2., label = 'English')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linestyle='--', linewidth = 2., label = 'Shared')
        plt.plot([-5,-4], [0.0, 0.0], marker = '+', color = 'none', label = '  ')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 3., label = '2xActivation')
        plt.plot([-5,-4], [0.0, 0.0], color = 'grey', linewidth = 1., label = '0xActivation')
        
        
        
        plt.xlim((min(ks), max(ks)))
        plt.xticks(ks, 
                   labels = config.ACCURACY_RANKS)
        plt.xscale('log')
        plt.xlabel('k')
        plt.ylabel(r'$\Delta$CCP@k')
        plt.title(f'CCP@k on T-REX - Without & Doubling KNs\n On Trivial Prompt "X [MASK] ."\n p = {kwargs["p_thresh"]} - {kwargs["dataset_name"]} - {scores[1]["kns_mode"]} - {np.round(scores[1]["threshold"],2)}')
        
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right margin to leave space for the legend

        # Create a single legend for all plots on the right side of the figure
        fig.legend(loc='center right', bbox_to_anchor=(0.99, 0.5))
        
        """
        plt.plot(np.arange(4), [scores[2]['vanilla'][k] for k in ccp_at_ks], marker = '+', linewidth = 3., color='black', label = 'vanilla')
        plt.plot(np.arange(4), [scores[2]['sem_wo_kns'][k] for k in ccp_at_ks], color = 'blue', linestyle = '--', marker = '+', label = ' w/o Sem KNs')
        plt.plot(np.arange(4), [scores[2]['sem_db_kns'][k] for k in ccp_at_ks], color = 'blue', marker = '+', label = 'db Sem KNs')
        plt.plot(np.arange(4), [scores[2]['know_wo_kns'][k] for k in ccp_at_ks], color = 'green', linestyle = '--', marker = '+', label = ' w/o Know KNs')
        plt.plot(np.arange(4), [scores[2]['know_db_kns'][k] for k in ccp_at_ks], color = 'green', marker = '+', label = 'db Know KNs')
        plt.plot(np.arange(4), [scores[2]['syn_wo_kns'][k] for k in ccp_at_ks], color = 'red', linestyle = '--', marker = '+', label = ' w/o Syn KNs')
        plt.plot(np.arange(4), [scores[2]['syn_db_kns'][k] for k in ccp_at_ks], color = 'red', marker = '+', label = 'db Syn KNs')
        
        plt.ylim((0,0.1))
        plt.xticks(np.arange(4), 
                labels = config.ACCURACY_RANKS)
        plt.xlabel('k')
        plt.ylabel('CCP@k')
        plt.legend()
        plt.title(f'CCP@k on T-REX - Without & Doubling KNs Semantics & Knowledge KNs\n On Trivial Prompt "X [MASK] ."\n p = {kwargs["p_thresh"]} - {kwargs["dataset_name"]} - {scores[2]["kns_mode"]} - {np.round(scores[2]["threshold"],2)}')
        #plt.gcf().subplots_adjust(top=0.15)
        """

            
        # Save
        plt.savefig(
            os.path.join(
                kns_path,
                kwargs['dataset_name'],
                f"kns_exp_2_ccp_thresh_{int(scores[2]['threshold']*100)}_{scores[2]['kns_mode']}_{kwargs['dataset_name']}_db_fact_{scores[2]['db_fact']}_p_{kwargs['p_thresh']}.png"
            )
        )
        plt.close()
    
    
    
##############################################################################
#                              DEPRECIATED                                   # 
##############################################################################
    
def plot_KNs_layer_distribution(layer_kns: Dict[str,int], **kwargs) -> None:

    num_neurons = 3072
    layer_scores = [0]*kwargs['num_layers']
    for layer, count in layer_kns.items():
        layer_scores[layer] += count

    layer_scores = [s*100/num_neurons for s in layer_scores]

    plt.bar(np.arange(len(layer_scores)), layer_scores)
    plt.xlabel('Layer')
    plt.ylabel('Percentage')
    if kwargs['overlap']:
        plt.title(f'Knowledge Neurons Overlap Layer Distribution')
        plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                f"kns_overlap_layer_distrib.png"
            )
        )
    else:
        plt.title(f'Knowledge Neurons Layer Distribution ({kwargs["dataset"]})')
        plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset'],
                f"kns_layer_distrib.png"
            )
        )
    plt.close()
    
def plot_KNs_categories(res: Dict[float, Dict[str, Dict[int, float]]],
                        thresh: float,
                        **kwargs) -> None:

    _thresh = find_closest_elem(res.keys() ,thresh)
    
    sem_scores = res[_thresh]['sem']
    pararel_syn_scores = res[_thresh]['pararel_syn']
    autoprompt_syn_scores = res[_thresh]['autoprompt_syn']
    pararel_know_scores = res[_thresh]['pararel_know']
    autoprompt_know_scores = res[_thresh]['autoprompt_know']
    pararel_autoprompt_know_scores = res[_thresh]['pararel_autoprompt_know']

    width = 1/4
    xs = np.arange(len(sem_scores))
    
    ##
    plt.rcParams['hatch.color'] = 'lime'
    plt.rcParams['hatch.linewidth'] = 3.5

    # Sem
    plt.bar(xs, list(sem_scores.values()), width = width, color = 'blue', label = 'sem.')
    # Syn
    plt.bar(xs + width, list(pararel_syn_scores.values()), width = width, color = 'red', label = 'pararel syn.')
    plt.bar(xs + width, list(autoprompt_syn_scores.values()), width = width, bottom = list(pararel_syn_scores.values()), color = 'darkred', label = 'autoprompt syn.')
    # Know
    plt.bar(xs + 2*width, list(pararel_know_scores.values()), width = width, color = 'green', label = 'pararel know')
    plt.bar(xs + 2*width, list(pararel_autoprompt_know_scores.values()), width = width, 
            bottom = list(pararel_know_scores.values()), hatch = '//', color = 'green')
    plt.bar(xs + 2*width, list(autoprompt_know_scores.values()), width = width, 
            bottom = np.array(list(pararel_know_scores.values())) + np.array(list(pararel_autoprompt_know_scores.values())), color = 'lime', label = 'autoprompt know')
    
    plt.xlabel('Layer')
    plt.ylabel('proportion')
    plt.title(f'Semantics, Syntax & Knowledge KNs Layer Distribution (thresh = {np.round(thresh,2)})')
    plt.xticks(np.arange(len(xs)) + width, labels=xs)
    plt.legend()
    os.makedirs(
        os.path.join(
            kwargs['kns_path'],
            'sem_syn_know'), exist_ok=True)
    plt.savefig(
        os.path.join(
            kwargs['kns_path'],
            'sem_syn_know',
            f"kns_categories_layer_distrib_thresh_{int(thresh*100)}.png" # x100 is for sorting purpose in folder
        )
    )
    plt.close()
    
    ## Restore Params
    plt.rcParams['hatch.color'] = 'black'
    plt.rcParams['hatch.linewidth'] = 0.1
    
def plot_KNs_num_by_threshold(kns: Dict[float, List[Tuple[float, float]]], **kwargs) -> None:

    plt.bar(np.array(list(kns.keys())) + 1/(2*len(kns)), 
            [v for v in kns.values()], 
            width=1/len(kns))
    plt.ylabel('Proportion of KNs')
    plt.xlabel('Threhsold')
    plt.xlim((0,1))
    plt.title(f'Proportion of KNs depending on Threshold ' + kwargs["title_suffix"])
    plt.savefig(
        os.path.join(
            kwargs['kns_path'],
            f"prop_kns_by_thresh_{kwargs['file_name_suffix']}.png"
        )
    )
    plt.close()
    
    
def plot_sem_kns_to_rela_category(sem_kns2rela: Dict[float, Dict[Tuple[float, float], Tuple[List[str], List[str]]]], **kwargs) -> None:
    
    MIN_THRESH = 0.3
    
    CATEGORIES_COLORS = {
                'Continent': 'darkred',                             ## Geography
                'Country': 'firebrick',                             #
                'Country-City': 'indianred',                        #
                'Region-City': 'lightcoral',                        #
                'City': 'mistyrose',                                #
                'Language': 'lightsalmon',                          ## Language
                'Religion': 'darkorange',                           ## Religion                   
                'Profession': 'mediumseagreen',                     ## Jobs
                'Profession-Field': 'green',                        #
                'Company': 'darkgreen',                             #
                'Country-City-Company-Person': 'mediumaquamarine',  # 
                'Sport_Position': 'cyan',                           #
                'Organism': 'lightblue',                            #
                'Radio-TV': 'indigo',                               # Music
                'Music': 'darkviolet',                              #
                'Music_Label': 'mediumorchid',                      #
                'Music_Instrument': 'violet',                       #
                'Thing': 'orange',                                  ## Thing
                }
    
    CATEGORIES_COORDS = {
                'Continent': 17,                             
                'Country': 16,                             
                'Country-City': 15,                        
                'Region-City': 14,                        
                'City': 13,                                
                'Language': 12,                          
                'Religion': 11,                                            
                'Profession': 10,                     
                'Profession-Field': 9,                       
                'Company': 8,                             
                'Country-City-Company-Person': 7,  
                'Sport_Position': 6,
                'Organism': 5,
                'Radio-TV': 4,
                'Music': 3,                          
                'Music_Label': 2,                    
                'Music_Instrument': 1,               
                'Thing': 0,                          
                }
    
    thresholds = list(sem_kns2rela.keys())
    thresholds.sort()
    thresholds.reverse() # We'll start by the last neurons
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20,7))
    
    x = 0. # Initialize coordinates
    xticks, xlabels = [], []
    seen_kns = set()
    for thresh in thresholds:
        if thresh < MIN_THRESH:
            continue # I don't think it's useful to plots below
        
        _sem_kns = sem_kns2rela[thresh]
        
        for kn, val in _sem_kns.items():
            if kn in seen_kns:
                continue # We're not plotting the same KNs multiple time
            else:
                seen_kns.add(kn)
        
            _, cats = val 
            for cat in cats:
                ax.add_patch(
                        mpatches.Rectangle(
                                        xy=(x, CATEGORIES_COORDS[cat]), 
                                        width=1., 
                                        height=1., 
                                        color = CATEGORIES_COLORS[cat]
                                        )
                    )
            x -= 1. 
        ax.vlines(x = x, ymin = 0., ymax = 18, linestyle = '--', alpha = 0.5, color = 'black')
        
        xticks = [x] + xticks  
        xlabels = [np.round(thresh, 2)] + xlabels 
    
    ylabels = list(CATEGORIES_COORDS.keys())
    ylabels.reverse()
    ax.set_xticks(ticks = xticks, labels = xlabels, rotation = 45)
    ax.set_yticks(ticks = np.arange(len(CATEGORIES_COORDS)) + 0.5, 
                  labels = ylabels)    
    ax.set_xlim(x, 0.5)
    ax.set_ylim(0., 18)
    
    ax.set_xlabel('Semantics KNs Decision Thresholds')
    plt.title('Semantics KNs Associated T-REX Category')
    
    os.makedirs(
        os.path.join(
            kwargs['kns_path'],
            'sem_syn_know'), exist_ok=True)
    plt.savefig(
        os.path.join(
            kwargs['kns_path'],
            'sem_syn_know',
            f"sem_kns_trex_category.png"
        )
    )
    plt.close()
    
def plot_sem_syn_know_dist(res_array: np.ndarray, **kwargs) -> None:
            
    xs = np.arange(res_array.shape[0])
    
    plt.bar(xs, res_array[:,0], label = 'semantics')
    plt.bar(xs, res_array[:,1], bottom = res_array[:,0], label = 'syntax')
    plt.bar(xs, res_array[:,2], bottom = res_array[:,0] + res_array[:,1], label = 'knowledge')
    
    plt.legend()
    
    xtickslabels = [np.round(x,3) for x in np.linspace(0,1, len(xs))]
    plt.xticks(xs, labels=xtickslabels, rotation = 360-45)
    
    plt.xlabel('Semantics & Syntax KNs Decision Threshold')
    plt.ylabel('Count')
    
    plt.title(f"Semantcis, Syntax & Knowledge KNs distribution ({kwargs['file_name_suffix']})")
    
    plt.gcf().subplots_adjust(bottom=0.15)

    os.makedirs(
        os.path.join(
            kwargs['kns_path'],
            'sem_syn_know'), exist_ok=True)
    plt.savefig(
        os.path.join(
            kwargs['kns_path'],
            'sem_syn_know',
            f"sem_syn_know_dist_{kwargs['file_name_suffix']}.png"
        )
    )
    plt.close()
    

        
        
        