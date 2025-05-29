from XAI.p4_on_mm_layers import P4
from XAI.visualization import visualize_cam
# from PIL import Image
# import matplotlib.pyplot as plt

dataloader = []

def explain():
    model = P4()
    model.eval()

    for x, label in dataloader:
        y = model(x)
        heatmap = model.explain(y, class_index=label, class_num=40)
        heatmap = visualize_cam(heatmap)  # [224,224]


        # fig, axs = plt.subplots(3, 1)
        # plt.subplots_adjust(wspace=0.03, hspace=0.03)
        # axs[0].imshow(heatmap)
        # axs[0].axis('off')
        # # axs[0].set_title(title + '-' + name_dict[true_idx])
        # # axs[0].set_title('Chefer' + title)
        # axs[1].imshow(heatmap)
        # axs[1].axis('off')
        # # axs[1].set_title('mm24')
        # axs[2].imshow(vis2)
        # axs[2].axis('off')
        # # axs[2].set_title('ours')

        # if platform.system() == 'Windows':
        #     save_name = f.split('\\')[-1].replace('.JPEG', '_')
        # else:
        #     save_name = f.split('/')[-1].replace('.JPEG', '_')
        # plt.savefig('../data/output_vit/' + save_name + ".jpg", dpi=500)
        # plt.clf()
        # plt.close('all')

