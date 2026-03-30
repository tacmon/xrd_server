import os
import sys

# Set working directory to project root for easy path access
# Get the absolute path of the directory containing this script (src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Novel-Space/)
base_dir = os.path.dirname(script_dir)
# Change the current working directory to Novel-Space/
os.chdir(base_dir)
# Add the project root to sys.path so autoXRD can be imported
root_dir = os.path.dirname(base_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from autoXRD import spectrum_generation, solid_solns, tabulate_cifs
# Use PyTorch CNN implementation
from autoXRD.cnn.pytorch_models import main as cnn_main
import numpy as np
import pymatgen as mg


if __name__ == '__main__':

    max_texture = 0.5 # default: texture associated with up to +/- 50% changes in peak intensities
    min_domain_size, max_domain_size = 5.0, 30.0 # default: domain sizes ranging from 5 to 30 nm
    max_strain = 0.03 # default: up to +/- 3% strain
    max_shift = 0.5 # default: up to +/- 0.5 degrees shift in two-theta
    impur_amt = 70.0 # Max amount of impurity phases to include (%)
    num_spectra = 50 # Number of spectra to simulate per phase
    separate = False # If False: apply all artifacts simultaneously
    min_angle, max_angle = 20.0, 60.0
    num_epochs = 50
    skip_filter = False
    include_elems = True
    enforce_order = False
    oxi_filter = False
    max_cpu = 8
    for arg in sys.argv:
        if '--max_texture' in arg:
            max_texture = float(arg.split('=')[1])
        if '--min_domain_size' in arg:
            min_domain_size = float(arg.split('=')[1])
        if '--max_domain_size' in arg:
            max_domain_size = float(arg.split('=')[1])
        if '--max_strain' in arg:
            max_strain = float(arg.split('=')[1])
        if '--max_shift' in arg:
            max_shift = float(arg.split('=')[1])
        if '--impur_amt' in arg:
            impur_amt = float(arg.split('=')[1])
        if '--num_spectra' in arg:
            num_spectra = int(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--num_epochs' in arg:
            num_epochs = int(arg.split('=')[1])
        if '--skip_filter' in arg:
            skip_filter = True
        if '--ignore_elems' in arg:
            include_elems = False
        if '--enforce_order' in arg:
            enforce_order = True
        if '--oxi_filter' in arg:
            oxi_filter = True
        if '--separate_artifacts' in arg:
            separate = True
        if '--max_cpu' in arg:
            max_cpu = int(arg.split('=')[1])

    if not skip_filter:
        # Filter CIF files to create unique reference phases
        assert 'All_CIFs' in os.listdir('.'), 'No All_CIFs directory was provided. Please create or use --skip_filter'
        assert 'References' not in os.listdir('.'), 'References directory already exists. Please remove or use --skip_filter'
        tabulate_cifs.main('All_CIFs', 'References', oxi_filter, include_elems, enforce_order)

    else:
        assert 'References' in os.listdir('.'), '--skip_filter was specified, but no References directory was provided'

    if '--include_ns' in sys.argv:
        # Generate hypothetical solid solutions
        solid_solns.main('References')

    # Load or simulate augmented XRD spectra
    if '--load' in sys.argv and os.path.exists('XRD.npy'):
        print("Loading pre-generated XRD spectra from XRD.npy...")
        xrd_specs = np.load('XRD.npy', allow_pickle=True).tolist()
    else:
        # Simulate augmented XRD spectra
        print("Generating virtual XRD spectra...")
        xrd_obj = spectrum_generation.SpectraGenerator('References', num_spectra, max_texture, min_domain_size,
            max_domain_size, max_strain, max_shift, impur_amt, min_angle, max_angle, separate, is_pdf=False, max_cpu=max_cpu)
        xrd_specs = xrd_obj.augmented_spectra

        # Save XRD patterns if flag is specified
        if '--save' in sys.argv:
            np.save('XRD', np.array(xrd_specs))

    # Train, test, and save the CNN with PyTorch
    test_fraction = 0.2
    model_filename = 'Model.pth'  # Use PyTorch format

    use_dynamic = False
    if '--use_dynamic' in sys.argv:
        use_dynamic = True

    cnn_main(
        xrd_specs,
        num_epochs,
        test_fraction,
        is_pdf=False,
        fmodel=model_filename,
        use_dynamic=use_dynamic
    )
