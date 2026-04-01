# top goes over 'iterations', dispatches kernel
#   next loop loop splits array into equal chunks by core
#       loop for each thread running over local chunks
#            kernel