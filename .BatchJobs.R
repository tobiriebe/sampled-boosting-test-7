cluster.functions = makeClusterFunctionsSLURM(template.file = "lrz_serial.tmpl", 
    list.jobs.cmd = c("squeue", "-h", "-o %i", "-u $USER", "--clusters=hugemem", "| tail -n +2"))
staged.queries = TRUE
raise.warnings = FALSE
debug = FALSE
