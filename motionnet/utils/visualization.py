import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements

def visualize_histogram_distribution_trajectory_type(train_set):
    """
    train_set is a list with nb_scenes trajectories (15)
    train_set[i] is a list with 1 element : a dictionary :
        train_set[i][0] is a dictionary with the following keys :
            - scenario_id: ()
            - obj_trajs: (nb_agents, nb_timesteps, nb_features)
            - track_index_to_predict ()
            ...
            - map_polylines: (nb_polylines, nb_points, 2) ...
            - trajectory_type

    trajectory_type : {0, 1 ... 7}
    class TrajectoryType:
            STATIONARY = 0
            STRAIGHT = 1
            STRAIGHT_RIGHT = 2
            STRAIGHT_LEFT = 3
            RIGHT_U_TURN = 4
            RIGHT_TURN = 5
            LEFT_U_TURN = 6
            LEFT_TURN = 7

    Output : plt that shows a histogram of the distribution of the trajectory type
    """
    count_dict = {}
    
    trajectory_type = { 0: "stationary", 1: "straight", 2: "straight_right",
            3: "straight_left", 4: "right_u_turn", 5: "right_turn",
            6: "left_u_turn", 7: "left_turn" }
    
    for scene in train_set:
        traj = scene[0]['trajectory_type']
        if traj in count_dict:
            count_dict[traj] += 1
        else:
            count_dict[traj] = 1

    x = list(count_dict.keys())
    count_values = list(count_dict.values())
    total_count = sum(count_values)

    # Normalize the values
    normalized_values = [count / total_count for count in count_values]

    # Now, normalized_values contains the normalized counts
    y = normalized_values
    plt.bar(x, y)
    
    #on affiche TOUS les types de trajectoires
    plt.xticks(list(trajectory_type.keys()), list(trajectory_type.values()), fontsize=20)
    plt.xlabel('Trajectory type', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Distribution of trajectory types. Nb_total_scenes = %s' % len(train_set), fontsize=25)
    
 

    #afficher les graduations en grand
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)

    plt.show()

    return plt













def check_loaded_data(data, index=0):
    agents = np.concatenate([data['obj_trajs'][..., :2], data['obj_trajs_future_state'][..., :2]], axis=-2)
    map = data['map_polylines']

    agents = agents[index]
    map = map[index]
    ego_index = data['track_index_to_predict'][index]
    ego_agent = agents[ego_index]

    fig, ax = plt.subplots()

    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    # Function to draw lines with a validity check

    # Plot the map with mask check
    for lane in map:
        if lane[0, -3] in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if lane[i, -3] > 0:
                draw_line_with_mask(lane[i, :2], lane[i, -2:], color='grey', line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    # Draw trajectories for other agents
    for i in range(agents.shape[0]):
        draw_trajectory(agents[i], line_width=2)
    draw_trajectory(ego_agent, line_width=2, ego=True)
    # Set labels, limits, and other properties
    vis_range = 100
    # ax.legend()
    ax.set_xlim(-vis_range + 30, vis_range + 30)
    ax.set_ylim(-vis_range, vis_range)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # As defined in the common_utils.py file
    # traj_type = { 0: "stationary", 1: "straight", 2: "straight_right",
    #         3: "straight_left", 4: "right_u_turn", 5: "right_turn",
    #         6: "left_u_turn", 7: "left_turn" }
    #
    # kalman_2s, kalman_4s, kalman_6s = list(data["kalman_difficulty"][index])
    #
    # plt.title("%s -- Idx: %d -- Type: %s  -- kalman@(2s,4s,6s): %.1f %.1f %.1f" % (1, index, traj_type[data["trajectory_type"][0]], kalman_2s, kalman_4s, kalman_6s))
    # # Return the axes object
    # plt.show()

    # Return the PIL image
    return plt
    # return ax


def concatenate_images(images, rows, cols):
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(image_list, column_counts):
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = original_height * column_counts[0]  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new('RGB', (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image



def visualize_scene(the_scene):
    """
    same as visualize_batch but for a single scene so dont need to provide the draw_index
    the_scene : dictionary containing information about the scene
    - obj_trajs : (num_agents, num_timesteps, num_features)
    - map_polylines : (num_lanes, num_points, num_features)
    ...
    """

    def draw_line_with_mask(point1, point2, color, line_width=4):
        """
        point1 : (num_features,) : represente les features du point 1 a un temps t
        point2 : (num_features,) : represente les features du point 2 a un temps t

        #point1[0] : x
        #point1[1] : y
        """
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)
    
    def interpolate_color_ego(t, total_t):

        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)
    
    def draw_trajectory(trajectory, line_width, ego=False):
        """
        trajectory : (num_timesteps, num_features)
        """
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)

                #on verifie que les deux points sont valides
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    #trajectory[t] shape (num_features,) ==>
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)



    #draw_index is the index of the scene to draw in the batch (among the 61 scenes available in the batch)
    #########################
    map_lanes = the_scene['map_polylines']
    map_mask = the_scene['map_polylines_mask']
    past_traj = the_scene['obj_trajs']
    
    #ici, one ne prend que les 2 premieres dimensions de map_lanes qui sont les coordonnées x et y
    map_xy = map_lanes[..., :2]

    #map type est un one hot encoding qui represente le type de la voie
    map_type = map_lanes[..., 0, -20:]
    

    # draw map
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # Plot the map with mask check
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        # convert onehot to index
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                draw_line_with_mask(lane[i], lane[i + 1], color='grey', line_width=1.5)

    # draw past trajectory
    for idx, traj in enumerate(past_traj):
        ##traj (21,39) ==> 21 : temps, 39 : features
        draw_trajectory(traj, line_width=2)

    #draw trajectory of ego agent
    ego_index = the_scene['track_index_to_predict']
    ego_traj = past_traj[ego_index]

    draw_trajectory(ego_traj, line_width=2, ego=True)

    #recupere la traj de l'agent ego
    ego_traj = past_traj[ego_index] #shape (num_timesteps, num_features)
    #plot avec une croix ses 2 derniers points
    ax.plot(ego_traj[-1, 0], ego_traj[-1, 1], 'rx')

 

    plt.axis('off')
    plt.tight_layout()

    traj_type = { 0: "stationary", 1: "straight", 2: "straight_right",
            3: "straight_left", 4: "right_u_turn", 5: "right_turn",
            6: "left_u_turn", 7: "left_turn" }
    
    ego_traj_type = the_scene['trajectory_type'] #tensor(1)

    ego_traj_last_pose = the_scene['obj_trajs_last_pos'][ego_index] #shape (3,)
    ax.plot(ego_traj_last_pose[0], ego_traj_last_pose[1], 'yx')

    center_gt_trajs = the_scene['center_gt_trajs'][ego_index] #shape (3,)
    ax.plot(center_gt_trajs[0], center_gt_trajs[1], 'kx')

    type_traj = traj_type[ego_traj_type]

    plt.title("Type = %s, durée = %s, intial pose of ego = %.3f %.3f , end pose of ego = %.3f %.3f" % (type_traj, ego_traj.shape[0], ego_traj[0, 0], ego_traj[0, 1], ego_traj[-1, 0], ego_traj[-1, 1]), fontsize=11)
    return plt



def visualize_batch(batch, draw_index=0):
    """
    only visualize the map and the trajectories in the batch
    """


    def draw_line_with_mask(point1, point2, color, line_width=4):
        """
        point1 : (num_features,) : represente les features du point 1 a un temps t
        point2 : (num_features,) : represente les features du point 2 a un temps t

        #point1[0] : x
        #point1[1] : y
        """
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)
    
    def interpolate_color_ego(t, total_t):

        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)
    
    def draw_trajectory(trajectory, line_width, ego=False):
        """
        trajectory : (num_timesteps, num_features)
        """
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)

                #on verifie que les deux points sont valides
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    #trajectory[t] shape (num_features,) ==>
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)


    batch = batch['input_dict']
    #batch[input_dict] is a dictionary that information about 61 sceness

    #draw_index is the index of the scene to draw in the batch (among the 61 scenes available in the batch)
    #########################
    map_lanes = batch['map_polylines'][draw_index].cpu().numpy()
    map_mask = batch['map_polylines_mask'][draw_index].cpu().numpy()
    past_traj = batch['obj_trajs'][draw_index].cpu().numpy() #shape (num_agents, num_timesteps, num_features)




    #ici, one ne prend que les 2 premieres dimensions de map_lanes qui sont les coordonnées x et y
    map_xy = map_lanes[..., :2]

    #map type est un one hot encoding qui represente le type de la voie
    map_type = map_lanes[..., 0, -20:]
    
    
    # draw map
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # Plot the map with mask check
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        # convert onehot to index
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                draw_line_with_mask(lane[i], lane[i + 1], color='grey', line_width=1.5)

    # draw past trajectory
    for idx, traj in enumerate(past_traj):
        ##traj (21,39) ==> 21 : temps, 39 : features
        draw_trajectory(traj, line_width=2)

    #draw trajectory of ego agent
    ego_index = batch['track_index_to_predict'][draw_index]
    ego_traj = past_traj[ego_index]
    draw_trajectory(ego_traj, line_width=2, ego=True)

    #recupere la traj de lagent ego
    ego_traj = past_traj[ego_index] #shape (num_timesteps, num_features)
    #plot avec une croix ses 2 derniers points
    ax.plot(ego_traj[-1, 0], ego_traj[-1, 1], 'rx')
    




    #print les features disponible pour chaque agent : obj_trajs has shape (batch_size, num_agents, num_timesteps, num_features)
    #on va essayé de print les features pour chaque agent


    plt.axis('off')
    plt.tight_layout()

    #titre avec le type de la trajectoire
    traj_type = { 0: "stationary", 1: "straight", 2: "straight_right",
            3: "straight_left", 4: "right_u_turn", 5: "right_turn",
            6: "left_u_turn", 7: "left_turn" }
    
    ego_traj_type = batch['trajectory_type'][draw_index] #tensor(1)

    ### utilise objs_trajs_last_pose pour affiche avec une croix jaune la derniere position de l'agent ego
    ego_traj_last_pose = batch['obj_trajs_last_pos'][draw_index][ego_index] #shape (3,)
    ax.plot(ego_traj_last_pose[0], ego_traj_last_pose[1], 'yx')
    #ego_traj_last_pose[2] represente l'angle de l'agent ego

    ## NORMALEMENT, obj_trajs_last_pos et le dernier xy de obj_trajs devraient etre les memes

    ### utilise center_gt_trajs pour affiche avec une croix noire la derniere position de l'agent ego
    center_gt_trajs = batch['center_gt_trajs'][draw_index][ego_index] #shape (3,)
    ax.plot(center_gt_trajs[0], center_gt_trajs[1], 'kx')

    plt.title("Type = %s, durée = %s and intial pose of ego = %.3f %.3f " % (traj_type[ego_traj_type.item()], ego_traj.shape[0], ego_traj[0, 0], ego_traj[0, 1]), fontsize=16)

    #to plot with 2 decimals : %.2f
    #encadre le titre en rouge 
    ax.title.set_color('red')
    return plt










def visualize_prediction(batch, prediction, draw_index=0):
    """
    will visualize the map, past trajectory, future trajectory, and predicted future trajectory
    and plot them on the map
    """
    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    batch = batch['input_dict']
    map_lanes = batch['map_polylines'][draw_index].cpu().numpy()
    map_mask = batch['map_polylines_mask'][draw_index].cpu().numpy()
    past_traj = batch['obj_trajs'][draw_index].cpu().numpy()
    # future_traj = batch['obj_trajs_future_state'][draw_index].cpu().numpy()
    # past_traj_mask = batch['obj_trajs_mask'][draw_index].cpu().numpy()
    # future_traj_mask = batch['obj_trajs_future_mask'][draw_index].cpu().numpy()
    pred_future_prob = prediction['predicted_probability'][draw_index].detach().cpu().numpy()
    pred_future_traj = prediction['predicted_trajectory'][draw_index].detach().cpu().numpy()

    map_xy = map_lanes[..., :2]

    map_type = map_lanes[..., 0, -20:]

    # draw map
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # Plot the map with mask check
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        # convert onehot to index
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                draw_line_with_mask(lane[i], lane[i + 1], color='grey', line_width=1.5)

    # draw past trajectory
    for idx, traj in enumerate(past_traj):
        draw_trajectory(traj, line_width=2)

    # # draw future trajectory
    # for idx, traj in enumerate(future_traj):
    #     draw_trajectory(traj, line_width=2)

    # predicted future trajectory is (n,future_len,2) with n possible future trajectories, visualize all of them
    for idx, traj in enumerate(pred_future_traj):
        # calculate color based on probability
        color = cm.hot(pred_future_prob[idx])
        for i in range(len(traj) - 1):
            draw_line_with_mask(traj[i], traj[i + 1], color=color, line_width=2)
    plt.axis('off')
    plt.tight_layout()
    return plt
