import matplotlib
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import numpy as np
import math


matplotlib.use('Agg')

class GAF_New:


    def __init__(self) -> None:
        pass

    def __call__(self, scaled_series):

        scaled_series = np.where(scaled_series >= 1.0, 1.0, scaled_series)
        scaled_series = np.where(scaled_series <= -1.0, -1.0, scaled_series)

        phi = np.arccos(scaled_series)
        gaf = self.tabulate(phi, phi, self.cos_sum)

        return gaf

    
    def tabulate(self, x, y, f):
        return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

    def cos_sum(self, a, b):
        return (math.cos(a+b))

class TSTransformation:

    def __init__(self, root_dir, df, granularity):
        self.root_dir = root_dir
        self.df = df
        self.granularity = granularity

    def apply_edited_gaf_single_feature(self, feature, saving_params={}):
        """
            This function generates GAF images of features.
            The minimum and maximum required for generating these images are calculated based on the values in a time window
        """
        df = self.df
        root_dir = self.root_dir
        granularity = self.granularity

        if feature not in df.columns: 
            return "{feature} is not present in the dataset's columns".format(feature = feature)

        # gasf = GramianAngularField(image_size = granularity, method='summation')
        gaf = GAF_New()
        df_feature = df[[feature]]
        file_name = feature
        run, start_index = 0, -1
        while True:
            # start_index = run * granularity
            start_index += 1
            end_index = start_index + granularity

            try: 
                sample_values = df_feature[start_index:end_index]
            except:
                sample_values = df_feature[start_index:df_feature.index[-1]]

            if sample_values.size < granularity:
                break
            # img_gasf = gasf.fit_transform(sample_values.T)
            # img_gasf = img_gasf.reshape(granularity,granularity)
            img_gasf = gaf(sample_values)
            plt.figure(figsize=(1, 1), dpi=224);
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.imshow(img_gasf, cmap='gray', vmin=-1, vmax=1)
            plt.imshow(img_gasf, vmin=-1, vmax=1)
            if saving_params != {}: 
                if saving_params['file_name'] != "":
                    add_file_name = saving_params['file_name']
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig("{}/{}_{}_{}.png".format(root_dir, add_file_name, file_name,str(run)),
                        bbox_inches = 'tight',pad_inches = 0)
                else:
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig("{}/{}_{}.png".format(root_dir, file_name,str(run)),
                        bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            run += 1
    
    def apply_gaf_single_feature(self, feature, saving_params={}):
        """
            This function generates GAF images of features.
            The minimum and maximum required for generating these images are calculated based on the values in a time window
        """
        df = self.df
        root_dir = self.root_dir
        granularity = self.granularity

        if feature not in df.columns: 
            return "{feature} is not present in the dataset's columns".format(feature = feature)

        gasf = GramianAngularField(image_size = granularity, method='summation')
        # gasf = GramianAngularField(image_size = granularity, method='difference')

        df_feature = df[[feature]]
        file_name = feature
        run, start_index = 0, 0
        while True:
            start_index += 1
            end_index = start_index + granularity

            try: 
                sample_values = df_feature[start_index:end_index]
            except:
                sample_values = df_feature[start_index:df_feature.index[-1]]

            if sample_values.size < granularity:
                break
            img_gasf = gasf.fit_transform(sample_values.T)
            img_gasf = img_gasf.reshape(granularity,granularity)
            plt.figure(figsize=(1, 1), dpi=224);
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(img_gasf, vmin=-1, vmax=1)
            if saving_params != {}: 
                if saving_params['file_name'] != "":
                    add_file_name = saving_params['file_name']
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig("{}/{}_{}_{}.png".format(root_dir, add_file_name, file_name,str(run)),
                        bbox_inches = 'tight',pad_inches = 0)
                else:
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig("{}/{}_{}.png".format(root_dir, file_name,str(run)),
                        bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            run += 1
    
    def partitioning_gaf_images(self, feature, save_images=False):
        """
        This function generates GAF images. Instead of calculating the min and max of each time window, this function 
        evaluates these values on the whole dataset. 
        As such, this function generates what we regard as `general` GAF images, which allows to reduce the effect of changes
        in a specific time window. 
        """
        df, root_dir, granularity, run = self.df, self.root_dir, self.granularity, 0

        if feature not in df.columns: 
            return "{feature} is not present in the dataset's columns".format(feature = feature)

        df_feature = df[[feature]]

        gasf = GramianAngularField(image_size=df_feature.shape[0],
        method = 'summation')

        img_gasf = gasf.fit_transform(df_feature.T)
        img_gasf = img_gasf.reshape(df_feature.shape[0], df_feature.shape[0])
        while True:
            start_index = run * granularity
            end_index = start_index + granularity

            try: 
                timestamps = img_gasf[start_index:end_index]
            except:
                break

            if len(timestamps) < granularity:
                break
            
            ovr_arr = np.array([])
        
            for row in timestamps:
                ovr_arr = np.append(ovr_arr, row[start_index:end_index])
                
            ovr_arr = ovr_arr.reshape(granularity, granularity) 
            plt.figure(figsize=(3, 3), dpi=96); # This command generates a 288x288 image
            plt.imshow(ovr_arr, cmap='gray', vmin=-1, vmax=1)
            
            if save_images==True: 
                plt.axis('off')
                plt.tight_layout()
                plt.savefig("{}/{}_{}.png".format(root_dir, feature, str(run)),
                bbox_inches = 0)
            plt.close();
            run += 1
        
    def combine_features(self, imgs_gasf, set_features, saving_params={}):
        """
            This function generates GAF images that combine 4 features, that are combined from left to right in a 2x2 grid. The generation process for each part of the grid
            is similar to the `partitioning_gaf_images` function, which grants the generated images a more holistic flair. 
        """
        root_dir, granularity = self.root_dir, self.granularity
        file_name = "_".join(set_features)
        run, end_index = 0, 0

        while end_index < (imgs_gasf[set_features[0]].shape[0] - granularity):
            fig_width = 3.  # inches
            fig_height = fig_width

            f, axarr = plt.subplots(2,2, figsize=(fig_width, fig_height), 
            gridspec_kw={'height_ratios':[1, 1]}, dpi=96)
            axarr = axarr.flatten()
            start_index = run * granularity
            end_index = start_index + granularity
            
            for idx, feature in enumerate(set_features):
                img_gasf = imgs_gasf[feature]
                
                timestamps = img_gasf[start_index:end_index]
                
                ovr_arr = np.array([])
        
                for row in timestamps:
                    ovr_arr = np.append(ovr_arr, row[start_index:end_index])
                    
                ovr_arr = ovr_arr.reshape(granularity, granularity)

                del img_gasf, over_arr
                axarr[idx].imshow(ovr_arr, cmap='gray', vmin=-1, vmax=1)
                axarr[idx].axis('off')
                plt.tight_layout()
                
                axarr[idx].set_aspect('equal')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            
            if saving_params != {}: 
                if saving_params['file_name'] != "":
                    add_file_name = saving_params['file_name']
                    plt.axis('off')
                    plt.savefig("{}/{}_{}_{}.png".format(root_dir, add_file_name, file_name,str(run)),
                        bbox_inches = 0)
                else:
                    plt.axis('off')
                    plt.savefig("{}/{}_{}.png".format(root_dir, file_name,str(run)),
                        bbox_inches = 0)
            plt.close();
            run += 1

    def combine_features_individual(self, df, set_features, saving_params = {}):
        """
            This function generates images that combine 4 features, each generated using the `apply_gaf_single_feature` function.
        """
        file_name = "_".join(set_features)
        root_dir, granularity, df = self.root_dir, self.granularity, self.df
        run, end_index = 0, 0

        while end_index < (df.shape[0] - granularity):
            fig_width = 3.  # inches
            fig_height = fig_width

            f, axarr = plt.subplots(2,2, figsize=(fig_width, fig_height), 
            gridspec_kw={'height_ratios':[1, 1]}, dpi=96)
            axarr = axarr.flatten()
            start_index = run * granularity
            end_index = start_index + granularity
            
            for idx, feature in enumerate(set_features):
                gasf = GramianAngularField(image_size = granularity, method='summation')
                df_feature = df[[feature]]

                start_index = run * granularity
                end_index = start_index + granularity

                try: 
                    sample_values = df_feature[start_index:end_index]
                except:
                    if end_index - start_index < granularity: 
                        break
                
                img_gasf = gasf.fit_transform(sample_values.T)
                img_gasf = img_gasf.reshape(granularity,granularity)
                
                axarr[idx].imshow(img_gasf, cmap='gray', vmin=-1, vmax=1)
                del df_feature, sample_values, img_gasf

                axarr[idx].axis('off')
                plt.tight_layout()
                
                axarr[idx].set_aspect('equal')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            
            if saving_params != {}: 
                if saving_params['file_name'] != "":
                    add_file_name = saving_params['file_name']
                    plt.axis('off')
                    plt.savefig("{}/{}_{}_{}.png".format(root_dir, add_file_name, file_name,str(run)),
                        bbox_inches = 0)
                else:
                    plt.axis('off')
                    plt.savefig("{}/{}_{}.png".format(root_dir, file_name,str(run)),
                        bbox_inches = 0)
            plt.close();
            run += 1