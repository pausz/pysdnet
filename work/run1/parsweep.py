    random.seed(42)
    ngrid         = 32
    vel, gsc, exc = logspace(-1, 2, ngrid), logspace(-5., -1.5, ngrid), r_[0.95:1.1:ngrid*1j]
    datasets      = map(data.dsi.load_dataset, range(9))
    models        = ['fhn_euler']
    nic           = 32
    dt            = 0.5
    tf            = 500
    ds            = 10
    j             = 0
    start_time    = time.time()
    launch_count  = 0

