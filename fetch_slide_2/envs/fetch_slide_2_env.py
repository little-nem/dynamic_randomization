import os
import numpy as np

from gym.utils import EzPickle
from gym.envs.robotics.fetch_env import FetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')

class FetchSlide2(FetchEnv, EzPickle):
    '''
    FetchSlide dependent on properties
    '''

    def __init__(self, reward_type='sparse', assets_file='slide.xml', eval_args=None):
        '''
        slide2.xml: deformable
        slide3.xml: normal but with surrounding box
        '''
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.32441906, 0.75018422, 0.5, 1., 0., 0., 0.],
        }
        self.max_angle = 25. / 180. * np.pi
        self.eval_args = eval_args
        FetchEnv.__init__(
            self, 'fetch/{}'.format(assets_file),
            has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        EzPickle.__init__(self)


    def _distance_constraint(self, goal):
        r =  np.abs(goal[0] - self.initial_gripper_xpos[0] - self.obj_range/2)
        theta = self.max_angle * (np.random.rand() - 0.5) * 2.0
        object_pos_g = np.array([-r * np.cos(theta), r * np.sin(theta)])
        object_pos = object_pos_g + goal[:2]
        return object_pos


    def _sample_goal(self):
        
        if not self.eval_args or self.eval_args['goal_eval'] == 'random':
            return super(FetchSlide2, self)._sample_goal()
        elif not self.eval_args or self.eval_args['goal_eval'] == 'oor-box':
            return self._out_of_reach_goal()
        else:
            goal = self._fixed_goal(self.eval_args['goal_pose'])
            if self.eval_args['start_eval'] == "constrained":
                self._constrained_start(goal)
            return goal


    def _fixed_goal(self, pose):

        table_size = self.get_property('table0', 'geom_size')
        table_pos = self.get_property('table0', 'body_pos')

        table_middle_u = table_pos + table_size
        table_middle_u[1] -= table_size[1]
        table_middle_u[0] -= self.target_range 

        goal = table_middle_u
        if pose:
            goal[0:2] = pose

        goal[2] = self.height_offset
        return goal.copy()


    def _out_of_reach_goal(self):

        try:
            b_low = float(self.eval_args['goal_pose'][0])
            b_range = float(self.eval_args['goal_pose'][1])
        except Exception:
            b_low = 0.6
            b_range = 0.1

        self.target_offset=np.array([b_low, 0.0, 0.0])  
        self.target_range=b_range  
        return super(FetchSlide2, self)._sample_goal()


    def _constrained_start(self, goal):
        # sample starting position of the object at radius of r from the goal        
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:2] = self._distance_constraint(goal)
        #object_qpos[:2] = np.array([1.12407974, 1.04765308])
        self.sim.forward()


    def object_ids(self, obj_name):
        obj_id = {}

        try:
            obj_id['body_id'] = self.sim.model.body_name2id(obj_name)
        except:
            print('Exception1')
            pass

        try:
            obj_id['geom_id'] = self.sim.model.geom_name2id(obj_name)
        except:
            print('Exception2')
            pass

        return obj_id


    def set_property(self, obj_name, prop_name, prop_value):
        obj_id = self.object_ids(obj_name)

        object_type = prop_name.split('_')[0]
        object_type_id = object_type + '_id'

        prop_id = obj_id[object_type_id]
        prop_all = getattr(self.sim.model, prop_name)
        # print('***',prop_name, object_type_id, prop_all[prop_id])#, prop_all[obj_id])

        prop_all[prop_id] = prop_value
        prop_all = getattr(self.sim.model, prop_name)


    def get_property(self, obj_name, prop_name):
        obj_id = self.object_ids(obj_name)

        object_type = prop_name.split('_')[0]
        object_type_id = object_type + '_id'

        prop_id = obj_id[object_type_id]
        prop_all = getattr(self.sim.model, prop_name)
        prop_val = prop_all[prop_id]
        return prop_val