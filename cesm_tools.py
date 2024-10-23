import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
import shutil
import os
import subprocess
import logging

rcg_logger = logging.Logger(__name__) # this should be replaced by a workflow utils


class RegionalCaseGen:

    def __init__(self):
        return

    def write_esmf_mesh(self, hgrid, bathymetry, save_path, title=None, cyclic_x=False):
        """
        Write the ESMF mesh file

        This function is adapted from Alper Altuntas' NCAR/mom6_bathy library.
        Modifications allow the function to work in isolation from the rest of the package.


        Parameters
        ----------
        hgrid: xarray dataset
            horizontal grid (supergrid) for MOM6
        ocean_mask: xarray dataset
            land/ocean mask for MOM6 run. 1 is ocean, 0 is land
        save_path: str
            Path to ESMF mesh file to be written.
        title: str, optional
            File title.
        """

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["gridType"] = "unstructured mesh"
        ds.attrs["date_created"] = datetime.now().isoformat()
        if title:
            ds.attrs["title"] = title

        tlon_flat = hgrid.x[
            1:-1:2, 1:-1:2
        ].values.flatten()  # Extract the T point cells from the hgrid
        tlat_flat = hgrid.y[1:-1:2, 1:-1:2].values.flatten()
        ncells = len(tlon_flat)  # i.e., elementCount in ESMF mesh nomenclature

        coord_units = (
            "degrees"  # Hardcode this to degrees rather than read in from supergrid
        )

        ds["centerCoords"] = xr.DataArray(
            [[tlon_flat[i], tlat_flat[i]] for i in range(ncells)],
            dims=["elementCount", "coordDim"],
            attrs={"units": coord_units},
        )

        ds["numElementConn"] = xr.DataArray(
            np.full(ncells, 4).astype(np.int8),
            dims=["elementCount"],
            attrs={"long_name": "Node indices that define the element connectivity"},
        )

        tarea = xr.DataArray(
            hgrid.area[::2, ::2]
            + hgrid.area[1::2, 1::2]
            + hgrid.area[::2, 1::2]
            + hgrid.area[::2, 1::2],
            dims=["ny", "nx"],
            attrs={"name": "area of t-cells", "units": "meters^2"},
        )

        ds["elementArea"] = xr.DataArray(
            tarea.values.flatten(),  # Read in the area from existing hgrid T cells
            dims=["elementCount"],
            attrs={
                "units": "m**2"
            },  # These units are hardcoded to match the regular mom6 hgrid
        )

        ocean_mask = xr.where(bathymetry.fillna(0) != 0, 1, 0)

        ds["elementMask"] = xr.DataArray(
            ocean_mask.values.astype(np.int32).flatten(), dims=["elementCount"]
        )

        i0 = 1  # start index for node id's

        if cyclic_x:

            nx, ny = len(hgrid.nx) // 2, len(hgrid.ny) // 2
            qlon_flat = hgrid.x[0:-1:2, 1:-1:2].values[:, :-1].flatten()
            qlat_flat = hgrid.y[0:-1:2, 1:-1:2].values[:, :-1].flatten()
            nnodes = len(qlon_flat)
            assert nnodes == nx * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            get_element_conn = lambda i: [
                i0 + i % nx + (i // nx) * (nx),
                i0 + i % nx + (i // nx) * (nx) + 1 - (((i + 1) % nx) == 0) * nx,
                i0 + i % nx + (i // nx + 1) * (nx) + 1 - (((i + 1) % nx) == 0) * nx,
                i0 + i % nx + (i // nx + 1) * (nx),
            ]

        else:  # non-cyclic grid

            nx, ny = len(hgrid.nx.values) // 2, len(hgrid.ny.values) // 2
            qlon_flat = hgrid.x[0::2, 0::2].values.flatten()
            qlat_flat = hgrid.y[0::2, 0::2].values.flatten()
            nnodes = len(qlon_flat)
            assert nnodes == (nx + 1) * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            get_element_conn = lambda i: [
                i0 + i % nx + (i // nx) * (nx + 1),
                i0 + i % nx + (i // nx) * (nx + 1) + 1,
                i0 + i % nx + (i // nx + 1) * (nx + 1) + 1,
                i0 + i % nx + (i // nx + 1) * (nx + 1),
            ]

        ds["nodeCoords"] = xr.DataArray(
            np.column_stack((qlon_flat, qlat_flat)),
            dims=["nodeCount", "coordDim"],
            attrs={"units": coord_units},
        )

        ds["elementConn"] = xr.DataArray(
            np.array([get_element_conn(i) for i in range(ncells)]).astype(np.int32),
            dims=["elementCount", "maxNodePElement"],
            attrs={
                "long_name": "Node indices that define the element connectivity",
                "start_index": np.int32(i0),
            },
        )
        ds.to_netcdf(save_path, mode="w")

    def setup_cesm(self, expt, CESMPath, project, cyclic_x=False):
        return self.setup_cesm_explicit(
            expt.hgrid,
            CESMPath,
            project,
            expt.mom_input_dir,
            expt.mom_run_dir,
            expt.date_range,
            expt.mom_input_dir / "bathymetry.nc",
            cyclic_x=cyclic_x,
        )

    def setup_cesm_explicit(
        self,
        hgrid,
        CESMPath,
        project,
        mom_input_dir,
        mom_run_dir,
        date_range,
        bathymetry_path,
        cyclic_x=False,
        cores_per_node=128,
        ideal_number_of_points_per_core_ceiling = 800,

    ):
        """
        Given a regional-mom6 experiment object and a path to the CESM folder, this function makes all of the changes to the CESM configuration to get it to run with the regional configuration.
        """

        nx = int(len(hgrid.nx) // 2)
        ny = int(len(hgrid.ny) // 2)
        # Copy the configuration files to the SourceMods folder
        rcg_logger.info(
            f"Copying input.nml, diag_table, MOM_input_and MOM_override to {CESMPath / 'SourceMods/src.mom'}"
        )
        for i in ["input.nml", "diag_table", "MOM_input", "MOM_override"]:
            shutil.copy(Path(mom_run_dir) / i, CESMPath / "SourceMods/src.mom")

        # Add NIGLOBAL and NJGLOBAL to MOM_override, and include INPUTDIR pointing to mom6 inputs - This can be replaced by the MOM_param functions once we have move it.
        rcg_logger.info(
            f"Adding NIGLOBAL = {nx}, NJGLOBAL = {ny}, and INPUTDIR = {mom_input_dir} to MOM_override"
        )
        with open(CESMPath / "SourceMods/src.mom/MOM_override", "a") as f:
            f.write(f"#override NIGLOBAL = {nx}\n")
            f.write(f"#override NJGLOBAL = {ny}\n")
            f.write(f"#override INPUTDIR = {mom_input_dir}\n")
            f.close()

        # Remove references to MOM_layout in input.nml, as processor layouts are handled by CESM
        rcg_logger.info("Add MOM_override to parameter_filename in input.nml")
        self.edit_input_nml_for_CESM(CESMPath, condition_strings=["MOM_layout", "parameter_filename"], new_string="parameter_filename = 'MOM_input', 'MOM_override'")

        # Move all of the forcing files out of the forcing directory to the main inputdir
        rcg_logger.info(
            "Move all of the forcing files out of the forcing directory to the main inputdir"
        )
        for i in mom_input_dir.glob("forcing/*"):
            shutil.move(i, mom_input_dir / i.name)

        # Find and replace instances of forcing/ with nothing in the MOM_input file - This can be replaced by the MOM_param functions once we have move it.
        rcg_logger.info(
            "Find and replace instances of forcing/ with nothing in the MOM_input file"
        )
        with open(CESMPath / "SourceMods/src.mom/MOM_override", "r") as f:
            lines = f.readlines()
            f.close()
        with open(CESMPath / "SourceMods/src.mom/MOM_override", "w") as f:
            for i in range(len(lines)):
                lines[i] = lines[i].replace("forcing/", "")
            f.writelines(lines)
            f.close()

        # Make ESMF grid and save to inputdir
        rcg_logger.info("Make ESMF grid and save to inputdir")
        self.write_esmf_mesh(
            hgrid,
            xr.open_dataarray(bathymetry_path),
            mom_input_dir / "esmf_mesh.nc",
            title="Regional MOM6 grid",
            cyclic_x=cyclic_x,
        )

        # Load Balancing Math
        total_number_of_points = nx * ny
        nodes=1
        pts_per_processor = total_number_of_points/ float(cores_per_node)
        while pts_per_processor > ideal_number_of_points_per_core_ceiling:
            nodes = nodes+1
            pts_per_processor = total_number_of_points / float(cores_per_node * nodes)
            
        
        # Avoid one node for all other components in ocean_only mode
        self.xmlchange(CESMPath, "ROOTPE_OCN", str(cores_per_node))
        # Set the number of processors
        self.xmlchange(CESMPath, "NTASKS_OCN", nodes*cores_per_node)


        # Make xml changes
        self.xmlchange(CESMPath, "OCN_NX", str(nx))
        self.xmlchange(CESMPath, "OCN_NY", str(ny))
        self.xmlchange(CESMPath, "MOM6_MEMORY_MODE", "dynamic_symmetric")
        self.xmlchange(CESMPath, "OCN_DOMAIN_MESH", str(mom_input_dir / "esmf_mesh.nc"))
        self.xmlchange(CESMPath, "ICE_DOMAIN_MESH", str(mom_input_dir / "esmf_mesh.nc"))
        self.xmlchange(CESMPath, "MASK_MESH", str(mom_input_dir / "esmf_mesh.nc"))
        self.xmlchange(CESMPath, "MASK_GRID", str(mom_input_dir / "esmf_mesh.nc"))
        self.xmlchange(CESMPath, "OCN_GRID", str(mom_input_dir / "esmf_mesh.nc"))
        self.xmlchange(CESMPath, "ICE_GRID", str(mom_input_dir / "esmf_mesh.nc"))
        self.xmlchange(CESMPath, "RUN_REFDATE", str(date_range[0].strftime("%Y-%m-%d")))
        self.xmlchange(
            CESMPath, "RUN_STARTDATE", str(date_range[0].strftime("%Y-%m-%d"))
        )
        self.xmlchange(CESMPath, "PROJECT", str(project))
        self.xmlchange(CESMPath, "CHARGE_ACCOUNT", str(project))

        # Now make symlinks from the CESM directory to the mom input directory and the CESM run directory
        rcg_logger.info(
            "Make symlinks from the CESM directory to the mom input directory and the CESM run directory"
        )
        with CESMPath / "mom_input_directory" as link:
            link.unlink(missing_ok=True)
            link.symlink_to(mom_input_dir)

        return


    def edit_input_nml_for_CESM(self, CESMPath, condition_strings: list = ["MOM_layout", "parameter_filename"], new_string: str = "parameter_filename = 'MOM_input', 'MOM_override'"):
        """
        Remove reference to condition_strings in input.nml and adds the new_strong. The only reason to take this out of the setup_cesm function is to remove direct file changes from the main function.
        Parameters
        ----------
        CESMPath : Path
            Path to the CESM directory
        condition_strings : list, optional
            The strings that we are searching if a line already has, by default ["MOM_layout", "parameter_filename"]
        new_string : str, optional
            The new string to replace that line with, by default "parameter_filename = 'MOM_input', 'MOM_override'"
        """
        rcg_logger.info("Removing reference to MOM_layout in input.nml and add MOM_override to parameter_filename in input.nml")
        with open(CESMPath / "SourceMods/src.mom/input.nml", "r") as f:
            lines = f.readlines()
            f.close()
        rcg_logger.info("")
        with open(CESMPath / "SourceMods/src.mom/input.nml", "w") as f:
            for i in range(len(lines)):
                if all(cond in lines[i]  for cond in condition_strings):
                    rcg_logger.info(f"Modifying line: {lines[i].strip()}")
                    lines[i] = new_string
            f.writelines(lines)

    def xmlchange(self, CESM_path, param_name, param_value):
        """
        Run the XML Change Script at the CESM_path arg with param_name and param_value
        Parameters
        ----------
        CESM_path : Path
            Path to the CESM directory
        param_name : str
            name of the parameter to change
        param_value : str
            value to change the parameter to
        Returns
        -------
        subprocess.CompletedProcess
            The result of the subprocess.run command
        """
        rcg_logger.info(f"XML Change: {param_name} to {param_value}!")
        return subprocess.run(
            f"./xmlchange {param_name}={param_value}",
            shell=True,
            cwd=str(CESM_path),
        )

