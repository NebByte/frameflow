"""
frameflow -- recover a film's periphery from its own footage, and prove it.

Every module here runs on its own as well as importing: `python -m
frameflow.triage clip.mp4` answers which shots can be widened, `frameflow.polish`
fixes a finished wall using nothing but that wall's own photography. The
research log in docs/ treats each as a standalone finding, which is why they are
peers rather than a hierarchy.

The one idea underneath all of them: a pixel on a 270-degree side wall was
either PHOTOGRAPHED or INVENTED, and the two must never be added together.
"""
__version__ = "0.4.0"
