from utils import config, geometry, misc, evaluate

def main():
    args = config.load_parser()
    
    key_frame_info = misc.load_keyframe_info(args)
    poses = misc.load_final_poses(args)
    
    if args.finetune:
        model = misc.load_model(args, 0)
        model.eval()
        
        if args.eval_mesh:
            geometry.extract_mesh(args, model)
        if args.eval_depth:
            evaluate.gen_render_result(args, model, poses, key_frame_info)
            # evaluate.gen_render_result(args, model, gt_poses, key_frame_info)
        
        if args.pseudo_lidar:
            if args.eval_mesh:
                evaluate.eval_3D_mesh(args)
            if args.eval_depth:
                evaluate.eval_2D_depth_map(args)
            if args.eval_pose:
                gt_poses = misc.load_keyframe_gt_pose(args)
                evaluate.eval_pose_refinement(args, gt_poses, poses)
        
    else:
        geometry.save_pcd_bbox(args, poses, key_frame_info)
    
if __name__ == "__main__":
    main()
    