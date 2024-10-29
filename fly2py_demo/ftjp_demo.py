#importing module
import fly2py as ftjp



trx = ftjp.struct2df('trx.mat')


trx.extract_trx_param(['x', 'y', 'x_mm', 'y_mm', 'dt'], savefile=False)
trx.plot_density(plottype="heatmap", showplot=True, saveplot=True)
trx.plot_density(plottype="heatmap", showplot=True, burnin=20000)

trx.plot_tracks(bysex=False, saveplot=False, showplot=True)
trx.plot_tracks(bysex=True, saveplot=False, showplot=True)

dcenter = ftjp.struct2df('perframe/dcenter.mat')
dcenter.save_perframe_or_behavior("dcenter")



dcenter.plot_timeseries(showplot=True, saveplot=False)
dcenter.plot_timeseries(fly=4, showplot=True, saveplot=False, plottitle="fly 4 dcenter")


chase = ftjp.struct2df('scores_chase.mat')

chase.save_perframe_or_behavior()
chase.save_perframe_or_behavior(persecond=True)
chase.plot_timeseries(fly=6)



chase.plot_timeseries(showplot=True, saveplot=False)
chase.plot_timeseries(fly=[2,3,4,5], showplot=True, saveplot=False, plottitle="flies 2-5 chase")
chase.plot_timeseries(fly=[12,13,14,15], showplot=True, saveplot=False, plottitle="flies 12-15 chase")

wing = ftjp.struct2df('scores_wing_extension.mat')



ex = ftjp.fly_experiment([trx, dcenter, chase, wing])




ex.stack_timeseries(params="dcenter", behavior_scores=None, behavior_processed="chase", savefile=True)
ex.stack_timeseries(params="dcenter", behavior_scores=None, behavior_processed="chase", persecond=True, name="per_sec", savefile=True)


ex.ethogram(fly=3, showplot=True, saveplot=False, plottitle="fly 3")

ex.ethogram(fly=3, showplot=True, scorethreshold=0.9, saveplot=False, plottitle="fly 3, avg score>=0.9")
ex.ethogram(fly=3, burnin=500, showplot=True, scorethreshold=0.9, saveplot=False, plottitle="fly 3, 500s burnin, avg score>=0.9")
ex.ethogram(fly=[1,2,3,4,5], showplot=True, scorethreshold=0.9, saveplot=False, plottitle="fly 1-5, avg score>=0.9")



ex.network(dist_threshold=3, behavior='chase', behavior_threshold=1, showplot=True, saveplot=False, plottitle="chase")
ex.network(dist_threshold=3, behavior='wing_extension', behavior_threshold=1, showplot=True, saveplot=False, plottitle="wing_extension")


ex.network(behavior='chase', dist_threshold=1.5, behavior_threshold=1, showplot=True, saveplot=False, plottitle="chase network, dist<=1.5mm, score>=1")
ex.network(behavior='chase', dist_threshold=1.5, showplot=True, saveplot=False, plottitle="chase network, dist<=1.5mm, score>=0.5")

ex.network(behavior='chase', behavior_threshold=1, showplot=True, saveplot=False, plottitle="chase network, score>=1")



