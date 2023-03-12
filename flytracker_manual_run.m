% script for running flytracker from command line

cd %LOCAL/path/to/flytracker

addpath(genpath(pwd)); savepath;

cd %LOCAL/path/to/video/directory


videos.dir_in = '';
videos.dir_out = '';
videos.filter = '*.mp4';

options.num_cores = 10;
options.save_JAABA = 1;
options.granularity = 10000;




tracker(videos,options);




