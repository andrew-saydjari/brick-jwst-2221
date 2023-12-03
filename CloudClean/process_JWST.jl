## This is a pipeline to batch over JWST observations of The Brick
# Author - Andrew Saydjari, CfA

import Pkg; using Dates; t0 = now(); t_then = t0;
using InteractiveUtils; versioninfo()
Pkg.activate("/uufs/chpc.utah.edu/common/home/u6039752/scratch/julia_env/mamba_193_call"); #Pkg.instantiate(); Pkg.precompile()

t_now = now(); dt = Dates.canonicalize(Dates.CompoundPeriod(t_now-t_then)); println("Package activation took $dt"); t_then = t_now; flush(stdout)

using Distributed, SlurmClusterManager, Suppressor
# addprocs(SlurmManager())

t_now = now(); dt = Dates.canonicalize(Dates.CompoundPeriod(t_now-t_then)); println("Worker allocation took $dt"); t_then = t_now; flush(stdout)

activateout = @capture_out begin
    @everywhere begin
        import Pkg
        Pkg.activate("/uufs/chpc.utah.edu/common/home/u6039752/scratch/julia_env/mamba_193_call")        
    end
end

t_now = now(); dt = Dates.canonicalize(Dates.CompoundPeriod(t_now-t_then)); println("Worker activation took $dt"); t_then = t_now; flush(stdout)
 
@everywhere begin
    using FITSIO, StatsBase
    using CloudClean, ImageFiltering
    
    import PyPlot; const plt = PyPlot
    using PyCall
    mplcolors=pyimport("matplotlib.colors");
    mpltk=pyimport("mpl_toolkits.axes_grid1")
    patches=pyimport("matplotlib.patches")
    cc=pyimport("colorcet")
    pyimport("sys")."stdout" = PyTextIO(stdout)
    pyimport("sys")."stderr" = PyTextIO(stderr);
    plt.matplotlib.style.use("dark_background"); cc=pyimport("colorcet");
    
    using ProgressMeter
    using LinearAlgebra, BLISBLAS
    BLAS.set_num_threads(1)
end

t_now = now(); dt = Dates.canonicalize(Dates.CompoundPeriod(t_now-t_then)); println("Worker loading took $dt"); t_then = t_now; flush(stdout)

@everywhere begin
    cmapbkr = PyPlot.get_cmap("cet_bkr")
    cmapbkr.set_bad((0.1,0.1,0.1))

    cmap = PyPlot.get_cmap("cet_CET_L1_r")
    cmap.set_bad((0.7,0.1,0.1))

    nanmedian(x) = median(filter(!isnan,x));

    py"""
    import os
    os.environ['WEBBPSF_PATH']='/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_10_26/webbpsf-data'
    import webbpsf
    from webbpsf.utils import to_griddedpsfmodel
    import sys
    sys.path.append(os.path.abspath('/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_10_26/crowdwebb.py'))
    import crowdsource
    import crowdsource.crowdsource_base as cs
    import numpy as np
    from astropy.io import fits
    class WrappedPSFModel(crowdsource.psf.SimplePSF):
        def __init__(self, psfgridmodel):
            self.psfgridmodel = psfgridmodel
            self.default_stampsz = 39

        def __call__(self, col, row, stampsz=None, deriv=False):

            if stampsz is None:
                stampsz = self.default_stampsz

            parshape = np.broadcast(col, row).shape
            tparshape = parshape if len(parshape) > 0 else (1,)

            # numpy uses row, column notation
            rows, cols = np.indices((stampsz, stampsz)) - (np.array([stampsz, stampsz])-1)[:, None, None] / 2.
            # explicitly broadcast
            col = np.atleast_1d(col)
            row = np.atleast_1d(row)
            rows = rows[:, :, None] + row[None, None, :]
            cols = cols[:, :, None] + col[None, None, :]

            # photutils seems to use column, row notation
            # it returns something in (nstamps, row, col) shape
            # pretty sure that ought to be (col, row, nstamps) for crowdsource
            stamps = []
            for i in range(len(col)):
                stamps.append(self.psfgridmodel.evaluate(cols[:,:,i], rows[:,:,i], 1, col[i], row[i]))
            stampsS = np.stack(stamps,axis=0)
            #stamps = np.transpose(stampsS,axes=(0,2,1))
            stamps = stampsS

            if deriv:
                dpsfdrow, dpsfdcol = np.gradient(stamps, axis=(1,2))
                #dpsfdrow = dpsfdrow.T
                #dpsfdcol = dpsfdcol.T

            ret = stamps
            if parshape != tparshape:
                ret = ret.reshape(stampsz, stampsz)
                if deriv:
                    dpsfdrow = dpsfdrow.reshape(stampsz, stampsz)
                    dpsfdcol = dpsfdcol.reshape(stampsz, stampsz)
            if deriv:
                ret = (ret, dpsfdcol, dpsfdrow)

            return ret


        def render_model(self, col, row, stampsz=None):
            return None
    """

    function make_crowdsource_weight(im,err,wht)
        wt = 1 ./err
        mzerowt = (im .== 0) .| (err .== 0) .| (wht .== 0)
        mzerowt .|=  (isnan.(im)) .| (isnan.(err)) .| (isnan.(wht))
        mzerowt .|=  (err .< 1e-5)
        mzerowt .|=  (im .< percentile(im[.!mzerowt],[0.01])[1]) 
        wt[mzerowt].=0
        minw, maxw = percentile(filter(!iszero,wt),[1,99])
        msk = (wt.<minw) .& (wt .!=0)
        wt[msk].=minw
        msk = (wt.>maxw) .& (wt .!=0)
        wt[msk].=maxw
        return wt
    end

    function infill_JWST(filt)
        f = FITS("./data/jw02221-o001_t001_nircam_clear-$filt-merged-reproject_i2d.fits")
        obsdate = read_header(f["SCI"])["DATE-OBS"]
        im = read(f["SCI"])
        err = read(f["ERR"])
        wht = read(f["wht"]);
        close(f)

        filtupper = uppercase(filt)
        obsdatetime = obsdate*"T00:00:00"

        py"""
        nrc = webbpsf.NIRCam()
        obsdateT = $obsdatetime
        nrc.load_wss_opd_by_date(obsdateT)
        nrc.filter = $filtupper
        grid = nrc.psf_grid(num_psfs=16, all_detectors=True, verbose=True, save=True)
        temp = grid[0].copy()
        temp.fill_value = None
        psf_model = WrappedPSFModel(temp)
        """

        function load_psfmodel_cs()
            psfmodel_py = py"psf_model"
            function psfmodel_jl(x,y;stampsz=39,deriv=false)
                # accounts for x, y ordering and 0 v 1 indexing between python and Julia
                if length(x)==1
                    return psfmodel_py(y.-1,x.-1,stampsz=stampsz,deriv=deriv)
                else
                    # return permutedims(psfmodel_py(y.-1,x.-1,stampsz=stampsz,deriv=deriv),(1,3,2))
                    return psfmodel_py(y.-1,x.-1,stampsz=stampsz,deriv=deriv)
                end
            end
            return psfmodel_jl
        end

        psfmodel_py = load_psfmodel_cs()

        wt = make_crowdsource_weight(im,err,wht);
        d_im = ones(Int,size(wt))
        d_im[wt.==0].|= 2^1;
        im_clean = copy(im)
        im_clean[wt.==0].=0;

        resout = py"cs.fit_im"(im_clean,psfmodel_py,weight=wt,dq=d_im,
            ntilex=5, ntiley=2, refit_psf=false,psfderiv=true,fewstars=100,maxstars=320000,verbose=true,threshold=5,
            blendthreshu=0.2,psfvalsharpcutfac=0.7,psfsharpsat=0.7,miniter=2, maxiter=10, ccd=filtupper);

        rcat = resout[1]
        mod_im = resout[2]
        sky_im = resout[3]
        psf_out = resout[4];

        py"""
        hducat = fits.BinTableHDU($rcat)
        hducat.name = 'test'
        hdulist = fits.open("local_fit_$filt.fits", mode='append')
        hdulist.append(hducat)  # append the cat field for the ccd
        hdulist.close(closed=True)
        """

        f = FITS("local_fit_$(filt)_imgs.fits","w")
        write(f,im_clean,name="clean_im")
        write(f,mod_im,name="mod_im")
        write(f,sky_im,name="sky_im")
        write(f,wt,name="wt_im")
        close(f)

        # let me hold of on doing CloudClean here, I might want to interactively  mask, especially for 410m 
        # clean_im = im_clean
        # mod_im = mod_im
        # sky_im = sky_im
        # wt_im = wt
    end
end

filt_list = ["f466n", "f450n", "f410m" ]
@showprogress map(infill_JWST,filt_list)
# rmprocs(workers())

t_now = now(); dt = Dates.canonicalize(Dates.CompoundPeriod(t_now-t0)); println("Total script runtime: $dt"); t_then = t_now; flush(stdout)