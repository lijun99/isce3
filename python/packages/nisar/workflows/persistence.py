import os

import journal

class Persistence():
    '''
    basic class that determines InSAR persistence
    '''
    # init InSAR steps in reverse chronological run order. Must match the
    # forward order steps actually execute in insar.run(): bandpass_insar,
    # rdr2geo, geo2rdr, prepare_insar_hdf5, coarse_resample, dense_offsets,
    # offsets_product, rubbersheet, fine_resample, crossmul,
    # filter_interferogram, unwrap, ionosphere, geocode, troposphere,
    # solid_earth_tides, baseline.
    insar_steps = ['baseline', 'solid_earth_tides', 'troposphere', 'geocode', 'ionosphere', 'unwrap',
                   'filter_interferogram', 'crossmul', 'fine_resample', 'rubbersheet',
                   'offsets_product', 'dense_offsets', 'coarse_resample', 'prepare_insar_hdf5',
                   'geo2rdr', 'rdr2geo', 'h5_prep', 'bandpass_insar']

    def __init__(self, logfile_path, restart=False):
        """
        Construct a new `Persistence` object.

        Parameters
        ----------
        logfile_path : path_like
            Path to logfile from a previous InSAR workflow run.
        restart : bool, optional
            Whether to restart the workflow from the beginning or continue from a
            previous checkpoint. (default: False)
        """
        # bool flag that determines if insar.run is called
        # prevents calling of insar.run if last run was successful and no restart
        self.run = restart

        # dict key: step name data: bool for whether or not to run step
        # assume all steps successfully ran so default each steps run flag to false
        self.run_steps = {}
        for i in self.insar_steps:
            self.run_steps[i] = restart

        if not restart:
            self.read_log(logfile_path)

        info_channel = journal.info("persistence.init")
        if self.run:
            info_channel.log("Possible steps to be run:")
            for step in self.insar_steps:
                info_channel.log(f"{step}: {self.run_steps[step]}")
        else:
            info_channel.log("No steps to be (re)run.")

    def read_log(self, logfile_path):
        '''
        determine state of last run to determine this runs steps
        '''

        # check for empty log from error free runconfig and yamlparse execution
        if (not os.path.isfile(logfile_path)) or \
                (os.path.getsize(logfile_path) == 0):
            self.__init__(logfile_path, True)
            return

        # tracked across the whole scan (not reset per-line): whether we
        # found a usable success message to resume from
        success_msg_found = False

        # read log in reverse chronological order. The first (i.e. most
        # recent) "successfully ran ..." message we hit fully determines
        # the resume point, so we stop scanning once we find it.
        for log_line in reversed(list(open(logfile_path, 'r'))):
            # previous run completed the full workflow; nothing to rerun
            if 'successfully ran INSAR' in log_line:
                success_msg_found = True
                break

            # check for message indicating successful run of step
            if 'uccessfully ran' in log_line:
                # iterate thru reverse chronological steps
                for insar_step in self.insar_steps:
                    # any step not found in line will be step to run
                    if insar_step not in log_line:
                        # set step name found to True
                        self.run_steps[insar_step] = True
                        success_msg_found = True
                    else:
                        # check if any steps need to be run
                        if any(self.run_steps.values()):
                            self.run = True

                        # all previous steps successfully run and stop
                        break

                # the most recent success message fully determines the
                # resume point; older messages are redundant
                break

        # check if any steps need to be run or success msg not found
        if not success_msg_found:
            self.__init__(logfile_path, True)
