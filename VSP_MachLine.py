import os
import numpy as np
import openvsp as vsp
from matplotlib import pyplot as plt
import trimesh
import math
import re
import subprocess
import sys
import shutil


# define main geometry class
class VSP_Geometry:  
    # initialize class
    def __init__(self, casename="vspcase"):   
        pass 

    # define wing class
    class addWing:
        # initialize addWing class
        def __init__(self, sym=0):
            self.params = None  # placeholder for storing wing parameters
            self.airLwCoef = {'init': 1}  # lower surface coefficients dictionary
            self.airUpCoef = {'init': 1}  # upper surface coefficients dictionary
            self.symm = sym  # symmetry on

        def getParams(self, params):
            # check if chords key exists in params
            if "root_chords" not in params.keys():
                raise Exception('Wing param requires all necessary options!!')
            if "tip_chords" not in params.keys():
                raise Exception('Wing param requires all necessary options!!')
            # check if spans key exists in params
            if "spans" not in params.keys():
                raise Exception('Wing param requires all necessary options!!') 
            Ns = len(params["spans"])  
            paramsAll = params                

            # set default values for missing parameters
            if params.get("Tess_W") is None:  
                paramsAll["Tess_W"] = 33  # defalt to 33
            if params.get("xyzLoc") is None:  #
                paramsAll["xyzLoc"] = np.zeros((3,1)).tolist()  # default to (0,0,0)
            if params.get("xyzRot") is None:  
                paramsAll["xyzRot"] = np.zeros((3,1)).tolist()  # default to (0,0,0)
            if params.get("sweep") is None:  
                paramsAll["sweep"] = np.zeros((Ns)).tolist()  # default to 0 for all spans
            if params.get("sweepLoc") is None:  
                paramsAll["sweepLoc"] = np.zeros((Ns)).tolist()  # default to 0 for all spans
            if params.get("twist") is None:  
                paramsAll["twist"] = np.zeros((Ns)).tolist()  # default to 0 for all spans
            if params.get("twistLoc") is None:  
                paramsAll["twistLoc"] = np.ones((Ns)).tolist() * 0.25  # default to 0.25 for all spans
            if params.get("dihedral") is None: 
                paramsAll["dihedral"] = np.zeros((Ns)).tolist()  # default to 0 for all spans
            if params.get("Tess_U") is None:  
                paramsAll["Tess_U"] = np.ones((Ns)).tolist() * 10  # default to 10 for all spans
            if params.get("incidence") is None:  
                paramsAll["incidence"] = 0  # default to 0
            
            # store the final parameters in the class
            self.params = paramsAll

        # assign airfoil coefficients
        def appendAirfoil(self, aLw=[], aUp=[], sec_id=""):
            self.airLwCoef[sec_id] = aLw  # assign lower surface coefficients to the section id
            self.airUpCoef[sec_id] = aUp  # assign upper surface coefficients to the section id

        # read airfoil and fit using Bernstein
        def fitCSTAirfoil(self, Path, dp):
            x0, z0 = readAirfoil(Path)  # read airfoil (x and z coordinates)
            xLW, zLW, xUP, zUP = seperateSides(x0, z0)  # separate into lower and upper surfaces
            dTeLw = zLW[0]  # trailing edge thickness - lower surface
            dTeUp = zUP[-1]  # trailing edge thickness - upper surface
            # compute coefficients 
            aLw = self.Bernstein_MLS(xLW, zLW, dp, dTeLw) # lower
            aUp = self.Bernstein_MLS(xUP, zUP, dp, dTeUp) # upper
            return aLw, aUp  # return coeff

        # compute CST coefficients using MLS
        def Bernstein_MLS(self, x, y, dp, dTe):
            from math import factorial as FC  
            N = len(x)  # number of data pts
            b_1 = np.zeros(dp)  # Bernstein polynomial terms
            px_1 = np.zeros(dp)  # powers of x
            p1x_1 = np.zeros(dp)  # powers of (1-x)

            # compute Bernstein polynomial terms
            for i in range(dp):
                b_1[i] = FC(dp-1) / (FC(i) * FC(dp-i-1))  # compute binomial coefficient            
                px_1[i] = i + 0.5  # set power term for x
                p1x_1[i] = dp - i  # set power term for (1-x)

            # placeholders
            b_2 = np.zeros(dp)
            px_2 = np.zeros(dp)
            p1x_2 = np.zeros(dp)
            b_3 = np.zeros(dp)
            px_3 = np.zeros(dp)
            p1x_3 = np.zeros(dp)
            A = np.zeros((dp, dp))  # initialize coefficient matrix
            R = np.zeros(dp)  # initialize result vector

            # build and solve least squares
            for i in range(dp):
                b_2[i] = b_1[i]
                px_2[i] = px_1[i]
                p1x_2[i] = p1x_1[i]
                for j in range(dp):
                    b_3[j] = b_2[i] * b_1[j]  # compute higher-order terms
                    px_3[j] = px_2[i] + px_1[j]
                    p1x_3[j] = p1x_2[i] + p1x_1[j]            
                    for m in range(N):  
                        A[i, j] += ((x[m])**(px_3[j])) * ((1 - x[m])**(p1x_3[j])) * b_3[j]
                for m in range(N):  
                    R[i] += ((x[m])**(px_2[i])) * ((1 - x[m])**(p1x_2[i])) * b_2[i] * (y[m] - dTe * x[m])

            from scipy import linalg  
            X = linalg.solve(A, R)  
            return X 

    class addFuselage:
        def __init__(self):
            self.params = None
            self.sec_aLw = None
            self.sec_aUp = None
            self.secScales = None
            self.sec_zCnt = None
            self.skin_up = None
            self.skin_lw = None
            self.skin_sd = None
            self.aircraft = {}
            self.length = None

        def fit_fuselage_ellipse(self,stlPath,N_slice,dp):
            """fig = plt.figure()
            ax = plt.axes(projection='3d')"""
            mesh = trimesh.load(stlPath)
            min_x = mesh.bounds[0, 0]
            max_x = mesh.bounds[1, 0]    
            x_range = max_x - min_x
            xs = np.linspace(min_x, max_x, N_slice)
            xs[0] = xs[0]+1e-4
            xs[-1] = xs[-1]-1e-4
            self.sec_xs = xs
            self.params["XSec_Loc"] = ((xs - min_x) / (x_range)).tolist()

            lines_to_write = []
            secScales = []
            sec_aLw = np.zeros((N_slice,dp))
            sec_aUp = np.zeros((N_slice,dp))
            sec_zCnt = []
            sec_X_coor = np.zeros(N_slice)
            sec_Y_coor = np.zeros((N_slice,2))
            sec_Z_coor = np.zeros((N_slice,2))
            self.params["width"] = []
            self.params["height"] = []
            for iSec, x in enumerate(xs, start=0):
                slice_section = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
                if slice_section is None:
                    # No intersection found at this x
                    lines_to_write.append(f"Slice {iSec} at x={x:.5f} -> No intersection\n") 
                    # calculate chord lengths from width
                    continue
                width = (max(slice_section.vertices[:,1]) - min(slice_section.vertices[:,1]))*2
                height = (max(slice_section.vertices[:,2]) - min(slice_section.vertices[:,2]))                

                #ellipse parameters:
                self.params["width"].append(width/1000)#/1000            
                self.params["height"].append(height/1000)#/1000             

                # Slice the mesh    
                sec_X_coor[iSec] = x
                sec_Y_coor[iSec,0] = min(slice_section.vertices[:,1])
                sec_Y_coor[iSec,1] = max(slice_section.vertices[:,1])
                sec_Z_coor[iSec,0] = min(slice_section.vertices[:,2])
                sec_Z_coor[iSec,1] = max(slice_section.vertices[:,2])   
                self.params["ZSec_Loc"].append(( (sec_Z_coor[iSec,0]+sec_Z_coor[iSec,1])/2 )/(x_range))          

            m_up = np.zeros(N_slice*2-2)
            m_lw = np.zeros(N_slice*2-2)
            m_sd = np.zeros(N_slice*2-2)
            m_idx = 0       
            for i in range(N_slice-1):
                m_up[m_idx] = (sec_Z_coor[i+1,1]-sec_Z_coor[i,1])/(sec_X_coor[i+1]-sec_X_coor[i])
                m_lw[m_idx] = (sec_Z_coor[i+1,0]-sec_Z_coor[i,0])/(sec_X_coor[i+1]-sec_X_coor[i])
                m_sd[m_idx] = (sec_Y_coor[i+1,1]-sec_Y_coor[i,1])/(sec_X_coor[i+1]-sec_X_coor[i])
                m_idx += 1
                m_up[m_idx] = (sec_Z_coor[i+1,1]-sec_Z_coor[i,1])/(sec_X_coor[i+1]-sec_X_coor[i])
                m_lw[m_idx] = (sec_Z_coor[i+1,0]-sec_Z_coor[i,0])/(sec_X_coor[i+1]-sec_X_coor[i])
                m_sd[m_idx] = (sec_Y_coor[i+1,1]-sec_Y_coor[i,1])/(sec_X_coor[i+1]-sec_X_coor[i])                
                m_idx += 1      

            self.skin_up = np.atan(m_up)*(180/math.pi) # skinning angle upper
            self.skin_lw = np.atan(m_lw)*(180/math.pi) # skinning angle lower
            self.skin_sd = np.atan(m_sd)*(180/math.pi) # skinning angle side (R-L symmetry, almost always)
            self.length = (sec_X_coor[-1]-sec_X_coor[0])/1000

    class buildVSP:
        def __init__(self,aircraft,vspFolderPath,casename):
        
            self.vspFolderPath = vspFolderPath  
            self.casename  = casename            
            self.aircraft = aircraft
            self.defineWings()   
            self.defineFuselage()
            self.export_stl()

        def defineWings(self):
            list_of_comp = list(self.aircraft.keys())
            wing_idx = 0

            for comp in list_of_comp:
                if "wing" in comp:
                    wing = self.aircraft["wing" + str(wing_idx)]
                    wing_id = vsp.AddGeom("WING")                    
                    vsp.SetParmVal(wing_id, "X_Rel_Location", "XForm", wing.params["xyzLoc"][0])
                    vsp.SetParmVal(wing_id, "Y_Rel_Location", "XForm", wing.params["xyzLoc"][1])
                    vsp.SetParmVal(wing_id, "Z_Rel_Location", "XForm", wing.params["xyzLoc"][2])
                    vsp.SetParmVal(wing_id, "X_Rel_Rotation", "XForm", wing.params["xyzRot"][0])
                    vsp.SetParmVal(wing_id, "Y_Rel_Rotation", "XForm", wing.params["xyzRot"][1])
                    vsp.SetParmVal(wing_id, "Z_Rel_Rotation", "XForm", wing.params["xyzRot"][2])
                    vsp.SetParmVal(wing_id, "Tess_W", "Shape", wing.params["Tess_W"])
                    vsp.Update()
                    # Determine the number of sections
                    N_sect = len(wing.params["spans"])
                    for iSec in range(N_sect - 1):
                        vsp.InsertXSec(wing_id, iSec + 1, vsp.XS_CST_AIRFOIL)
                        vsp.UpdateGeom(wing_id)
                    vsp.Update()
                    for iSec in range(0, N_sect):
                        sec_name = f"XSec_{iSec+1}"                          
                        print(sec_name)
                        try:
                            vsp.SetParmVal(wing_id, "SectTess_U", sec_name, wing.params["Tess_U"][iSec]);vsp.UpdateGeom(wing_id)
                            vsp.SetParmVal(wing_id, "Root_Chord", sec_name, wing.params["root_chords"][iSec]);vsp.UpdateGeom(wing_id)
                            vsp.SetParmVal(wing_id, "Tip_Chord", sec_name, wing.params["tip_chords"][iSec]);vsp.UpdateGeom(wing_id)
                            vsp.SetParmVal(wing_id, "Span", sec_name, wing.params["spans"][iSec]);vsp.UpdateGeom(wing_id)
                            vsp.SetParmVal(wing_id, "Sweep", sec_name, wing.params["sweep"][iSec]);vsp.UpdateGeom(wing_id)
                            #vsp.SetParmVal(wing_id, "Sweep_Location", sec_name, wing.params["sweepLoc"][iSec])
                            vsp.SetParmVal(wing_id, "Twist", sec_name, wing.params["twist"][iSec]);vsp.UpdateGeom(wing_id)
                            vsp.SetParmVal(wing_id, "Dihedral", sec_name, wing.params["dihedral"][iSec]);vsp.UpdateGeom(wing_id)
                        except:
                            print(f"Warning: Failed to set parameters for {sec_name}")
                    #vsp.SetParmVal(wing_id, "Tip_Chord", f"XSec_{N_sect-1}", wing.params["root_chords"][N_sect]);vsp.UpdateGeom(wing_id)
                    vsp.Update()

                    # Assign airfoil shapes
                    for i in range(N_sect + 1):
                        xsec_surf = vsp.GetXSecSurf(wing_id, i)
                        vsp.ChangeXSecShape(xsec_surf, i, vsp.XS_CST_AIRFOIL)
                        dp = len(wing.airLwCoef[str(i)]) - 1
                        xsec = vsp.GetXSec(xsec_surf, i)
                        vsp.SetUpperCST(xsec, dp, wing.airUpCoef[str(i)])
                        vsp.SetLowerCST(xsec, dp, wing.airLwCoef[str(i)])

                    vsp.UpdateGeom(wing_id)
                    vsp.SetVSP3FileName(self.vspFolderPath + "/" + self.casename + ".vsp3")
                    vsp.WriteVSPFile(vsp.GetVSPFileName(), vsp.SET_ALL)
                    wing_idx += 1                       


        def defineFuselage(self):

            list_of_comp = list(self.aircraft.keys())
            fuselage_idx = 0

            for comp in list_of_comp:
                if "fuselage" in comp:
                    fuselage = self.aircraft["fuselage"]
                    fuselage_id = vsp.AddGeom("FUSELAGE");vsp.UpdateGeom(fuselage_id)

                    vsp.SetParmVal(fuselage_id, "X_Rel_Location", "XForm", fuselage.params["xyzLoc"][0]);vsp.UpdateGeom(fuselage_id)
                    vsp.SetParmVal(fuselage_id, "Y_Rel_Location", "XForm", fuselage.params["xyzLoc"][1]);vsp.UpdateGeom(fuselage_id)
                    vsp.SetParmVal(fuselage_id, "Z_Rel_Location", "XForm", fuselage.params["xyzLoc"][2]);vsp.UpdateGeom(fuselage_id)
                    vsp.SetParmVal(fuselage_id, "X_Rel_Rotation", "XForm", fuselage.params["xyzRot"][0]);vsp.UpdateGeom(fuselage_id)
                    vsp.SetParmVal(fuselage_id, "Y_Rel_Rotation", "XForm", fuselage.params["xyzRot"][1]);vsp.UpdateGeom(fuselage_id)
                    vsp.SetParmVal(fuselage_id, "Z_Rel_Rotation", "XForm", fuselage.params["xyzRot"][2]);vsp.UpdateGeom(fuselage_id)
                    vsp.SetParmVal(fuselage_id, "Length", "Design", fuselage.length);vsp.UpdateGeom(fuselage_id)
                    vsp.SetParmVal(fuselage_id, "Tess_W", "Shape", fuselage.params["Tess_W"]);vsp.UpdateGeom(fuselage_id)
                    print(f"Setting parameters for: Fuselage")

                    # number of sections
                    N_sect = len(fuselage.params["SectTess_U"]) +1


                    for iSec in range(N_sect - 5):  # CST_AIRFOIL assigned to fuselage - not properly working
                        vsp.InsertXSec(fuselage_id, iSec+1, vsp.XS_ELLIPSE)
                        vsp.UpdateGeom(fuselage_id)
                        vsp.Update()


                    for iSec in range(N_sect+1 // 2):
                        iSec2 = N_sect-1-iSec
                        if iSec!=0:      
                                sec_name = f"XSec_{iSec}"                                                   
                                vsp.SetParmVal(fuselage_id, "SectTess_U", sec_name, fuselage.params["SectTess_U"][iSec-1]);vsp.UpdateGeom(fuselage_id)
                                vsp.SetParmVal(fuselage_id, "XLocPercent", sec_name, fuselage.params["XSec_Loc"][iSec]);vsp.UpdateGeom(fuselage_id)
                                #vsp.SetParmVal(fuselage_id, "YLocPercent", sec_name, fuselage.params["YSec_Loc"][iSec]);vsp.UpdateGeom(fuselage_id)
                                vsp.SetParmVal(fuselage_id, "ZLocPercent", sec_name, fuselage.params["ZSec_Loc"][iSec]);vsp.UpdateGeom(fuselage_id)
                                print(f"Setting parameters for: {sec_name}")
                                if iSec !=iSec2:
                                    sec_name = f"XSec_{iSec2}" 
                                    vsp.SetParmVal(fuselage_id, "SectTess_U", sec_name, fuselage.params["SectTess_U"][iSec2-1]);vsp.UpdateGeom(fuselage_id)
                                    vsp.SetParmVal(fuselage_id, "XLocPercent", sec_name, fuselage.params["XSec_Loc"][iSec2]);vsp.UpdateGeom(fuselage_id)
                                    #vsp.SetParmVal(fuselage_id, "YLocPercent", sec_name, fuselage.params["YSec_Loc"][iSec]);vsp.UpdateGeom(fuselage_id)
                                    vsp.SetParmVal(fuselage_id, "ZLocPercent", sec_name, fuselage.params["ZSec_Loc"][iSec2]);vsp.UpdateGeom(fuselage_id)
                                    print(f"Setting parameters for: {sec_name}")                                    

                        try:                            
                            """vsp.SetParmVal(fuselage_id, "XRotate", sec_name, fuselage.params["xyzRot"][0])
                            vsp.SetParmVal(fuselage_id, "YRotate", sec_name, fuselage.params["xyzRot"][1])
                            vsp.SetParmVal(fuselage_id, "ZRotate", sec_name, fuselage.params["xyzRot"][2])"""  

                        except:
                            print(f"Warning: Failed to set parameters for {sec_name}")

                    # assign section parameters         
                    for iSec in range(0, N_sect-1):
                        if iSec==0 or iSec==N_sect-1:
                            xsec_surf = vsp.GetXSecSurf(fuselage_id, iSec);vsp.UpdateGeom(fuselage_id)
                            vsp.ChangeXSecShape(xsec_surf, iSec, vsp.XS_POINT);vsp.UpdateGeom(fuselage_id)
                        else:
                            curve_name = f"XSecCurve_{iSec}"  
                            xsec_surf = vsp.GetXSecSurf(fuselage_id, iSec);vsp.UpdateGeom(fuselage_id)
                            vsp.ChangeXSecShape(xsec_surf, iSec, vsp.XS_ELLIPSE);vsp.UpdateGeom(fuselage_id)
                            xsec = vsp.GetXSec(xsec_surf, iSec);vsp.UpdateGeom(fuselage_id)
                            #vsp.SetParmVal(fuselage_id, "Chord", curve_name, fuselage.params["chords"][iSec]); vsp.UpdateGeom(fuselage_id)  
                            #vsp.SetParmVal(fuselage_id, "Chord", curve_name, fuselage.params["chords"][iSec]); vsp.UpdateGeom(fuselage_id)
                            #print(fuselage.params["width"][iSec])
                            vsp.SetParmVal(fuselage_id, "Ellipse_Width", curve_name, fuselage.params["width"][iSec]); vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "Ellipse_Height", curve_name, fuselage.params["height"][iSec]); vsp.UpdateGeom(fuselage_id)

                    for iSec in range(0, N_sect):
                        sec_name = f"XSec_{iSec}"
                        chevron_name = f"Chevron_{iSec}"
                        if iSec < len(fuselage.skin_up):
                            vsp.SetParmVal(fuselage_id, "TBSym", sec_name, 0);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "ContinuityRight", sec_name, 0);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "TopRAngleSet", sec_name, 1);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "TopLAngleSet", sec_name, 1);vsp.UpdateGeom(fuselage_id)

                            vsp.SetParmVal(fuselage_id, "TopLStrength", sec_name, 0);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "BottomLStrength", sec_name, 0);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "RightLStrength", sec_name, 0);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "TopRStrength", sec_name, 0);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "BottomRStrength", sec_name, 0);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "RightRStrength", sec_name, 0);vsp.UpdateGeom(fuselage_id)                            

                            vsp.SetParmVal(fuselage_id, "TopLAngle", sec_name, fuselage.skin_up[iSec]);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "BottomLAngle", sec_name, fuselage.skin_lw[iSec]);vsp.UpdateGeom(fuselage_id)
                            vsp.SetParmVal(fuselage_id, "RightLAngle", sec_name, fuselage.skin_sd[iSec]);vsp.UpdateGeom(fuselage_id)
                            #vsp.SetParmVal(fuselage_id, "LeftLAngle", sec_name, fuselage.skin_sd[iSec]);vsp.UpdateGeom(fuselage_id)

                    """paramNames = vsp.FindContainerParmIDs(xsec_surf)
                    for iParm in range(len(paramNames)):
                        print(vsp.GetParmDisplayGroupName(paramNames[iParm]),vsp.GetParmName(paramNames[iParm]),vsp.GetParmDescript(paramNames[iParm]))"""
                    vsp.UpdateGeom(fuselage_id)
                    vsp.SetVSP3FileName(self.vspFolderPath + "/" + self.casename + ".vsp3")
                    vsp.WriteVSPFile(vsp.GetVSPFileName(), vsp.SET_ALL)
                    fuselage_idx += 1   

        def export_stl(self):
            compGeom = "CompGeom"
            vsp.ExecAnalysis(compGeom)
            vsp.Update()
             
            export_name =self.casename+ ".stl"
            export_path = os.path.join(self.vspFolderPath, export_name)
            
            vsp.ExportFile(export_path ,vsp.SET_ALL,vsp.EXPORT_STL)
            vsp.Update()
            print(export_path)

def VSP_Init():
    vsp.VSPCheckSetup()
    vsp.VSPRenew()
    vsp.ClearVSPModel()
    vsp.DeleteAllResults()


def readAirfoil(air_path):
    """
        This function read airfoil coordinates from a file path
    """
    # Open the file
    with open(air_path, "r") as f:
        #Create empyt arrays for  x and y coordinates
        x, y = [], []

        # Read each line in iteration
        for row in f:                

            #Get the current line with splitting it by spaces                        
            row = str(row).split()

            #if values are float, then append
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))

            #if values are not float (airfoil name), then skip
            except ValueError:
                print("Airfoil file invalid line: "+str(row))     
    return x, y


def polyBernstein(c1,c2,a,x):
    from math import factorial as FC
    dp = len(a)
    y= 0
    for j in range(dp):
        y += a[j]*FC(dp-1)/(FC(j)*FC(dp-j-1))*((x)**(j+c1))*((1 - x)**(dp+c2-j-1))
    return y

def seperateSides(x0,z0):    

    # seperate lower and upper surface
    # initial points must start from x=1
    # be sorted in the counter-clockwise direction.

    N = len(x0)
    xLW = []; zLW = []
    xUP = []; zUP = []
    N_lw = 0
    for i in range(N):
        xLW.append(x0[i])
        zLW.append(z0[i])        
        if z0[i]*z0[i+1]<=0 and x0[i] < x0[i+1]:
            N_lw = i+1
            break
    N_up = N-N_lw
    if xLW[N_lw-1] != 0 and zLW[N_lw-1] != 0:
        xLW.append(0)
        zLW.append(0)
        N_lw += 1 
    if x0[N_lw] != 0 and z0[N_lw] != 0:
        xUP.append(0)
        zUP.append(0)
        N_up += 1
    for i in range(N_lw,N):
        xUP.append(x0[i])
        zUP.append(z0[i])
    return xLW, zLW, xUP, zUP

def Bernstein_MLS(x, y, dp, dTe):

    # CST with sharp nose term

    from math import factorial as FC
    from scipy import linalg

    N = len(x)
    b_1  = np.zeros(dp)
    px_1 = np.zeros(dp)
    p1x_1= np.zeros(dp)

    for i in range(dp):
        b_1[i] = FC(dp - 1) / (FC(i) * FC(dp - i - 1))            
        px_1[i] = i + 0.5
        p1x_1[i] = dp - i

    A = np.zeros((dp + 1, dp + 1))
    R = np.zeros(dp + 1)

    for i in range(dp):
        b_i = b_1[i]
        px_i = px_1[i]
        p1x_i = p1x_1[i]
        for j in range(dp):
            b_j = b_1[j]
            px_j = px_1[j]
            p1x_j = p1x_1[j]
            coeff = b_i * b_j
            exp_x = px_i + px_j
            exp_1x = p1x_i + p1x_j
            for m in range(N):
                A[i, j] += (x[m] ** exp_x) * ((1 - x[m]) ** exp_1x) * coeff
        for m in range(N):
            R[i] += (x[m] ** px_i) * ((1 - x[m]) ** p1x_i) * b_i * (y[m] - dTe * x[m])

    # Add last row and column for the nose slope term
    for i in range(dp):
        b_i = b_1[i]
        px_i = px_1[i]
        p1x_i = p1x_1[i]
        for m in range(N):
            A[i, dp] += (x[m] ** px_i) * ((1 - x[m]) ** p1x_i) * b_i * (x[m] ** (dp + 0.5)) * (1 - x[m])
            A[dp, i] = A[i, dp]  # symmetry

    for m in range(N):
        A[dp, dp] += (x[m] ** (dp + 0.5)) * ((1 - x[m]) ** 2)
        R[dp] += (x[m] ** (dp + 0.5)) * (1 - x[m]) * (y[m] - dTe * x[m])

    X = linalg.solve(A, R)
    return X


"""#def Bernstein_MLS(x,y,dp,dTe):
    #-------------------------
    # method of least squares 
    # to find Bernstein coefficients
    #-------------------------
    from math import factorial as FC
    N    = len(x)
    b_1  = np.zeros(dp)
    px_1 = np.zeros(dp)
    p1x_1= np.zeros(dp)
    for i in range(dp):
        b_1[i] = FC(dp-1)/(FC(i)*FC(dp-i-1))            
        px_1[i] = i+0.5
        p1x_1[i] = dp-i
    b_2 = np.zeros(dp);  px_2 = np.zeros(dp); p1x_2 = np.zeros(dp)
    b_3 = np.zeros(dp);  px_3 = np.zeros(dp); p1x_3 = np.zeros(dp)
    A = np.zeros((dp,dp))
    R = np.zeros(dp)
    for i in range(dp):
        b_2[i]   = b_1[i]
        px_2[i]  = px_1[i]
        p1x_2[i] = p1x_1[i]
        for j in range(dp):
            b_3[j]   = b_2[i]*b_1[j]
            px_3[j]  = px_2[i]+px_1[j]
            p1x_3[j] = p1x_2[i]+p1x_1[j]            
            for m in range(N):
                A[i,j] = A[i,j] + ((x[m])**(px_3[j])) * ((1 - x[m])**(p1x_3[j]))*b_3[j]           
        for m in range(N):
            R[i] = R[i] + ((x[m])**(px_2[i]))*((1 - x[m])**(p1x_2[i]))*b_2[i]*(y[m]-dTe*x[m])    
    from scipy import linalg
    X = linalg.solve(A,R)
    return X"""

def bernstein_basis(n, k, x):
    from math import comb
    """
    Bernstein basis polynomial of degree n, index k, at x.
    b_{k,n}(x) = C(n, k) * x^k * (1 - x)^(n - k)
    """
    return comb(n, k) * (x**k) * ((1 - x)**(n - k))

def fit_bernstein_polynomial(x_data, y_data, n):

    #Fits a Bernstein polynomial of degree n to the data (x_data, y_data) 
    #using least squares. Assumes x_data is scaled to [0, 1].

    #Args:
    #    x_data (array-like): 1D array of x-values (in [0, 1])
    #    y_data (array-like): 1D array of y-values
    #    n (int): The degree of the Bernstein polynomial

    #Returns:
    #    c (numpy.ndarray): Array of best-fit coefficients [c_0, c_1, ..., c_n]
    #    residuals (float): Sum of squared residuals

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # number of data points
    m = len(x_data)
    
    # construct design matrix M (size m x (n+1))
    M = np.zeros((m, n + 1))
    for i in range(m):
        for k in range(n + 1):
            M[i, k] = bernstein_basis(n, k, x_data[i])

    # solve linear least squares problem M * c = y_data
    c, residuals, rank, s = np.linalg.lstsq(M, y_data, rcond=None)

    # If there are more data points than unknowns, residuals is an array
    # with the sum of squared residuals. If not, it might be empty.
    if len(residuals) > 0:
        residuals = residuals[0]
    else:
        # If no residuals are returned, it means it's a perfect fit or
        # the system is underdetermined. compute SSE manually:
        y_pred = M @ c
        residuals = np.sum((y_data - y_pred)**2)

    return c, residuals

def bernstein_polynomial(x, c):

    #Evaluate the fitted Bernstein polynomial at x given coefficients c.
    #len(c) - 1 = n (the degree).

    n = len(c) - 1
    val = 0.0
    for k in range(n + 1):
        val += c[k] * bernstein_basis(n, k, x)
    return val

# Path definitions
casePath = os.path.dirname(os.path.realpath(__file__))
casePath_VSP         = os.path.join(casePath, "VSP")
casePath_VSP_Results = os.path.join(casePath_VSP, "Results")
casePath_geometries  = os.path.join(casePath_VSP, "Geometries")
casePath_VSPAero     = os.path.join(casePath_VSP_Results, "VSPAero")
casePath_WaveDrag    = os.path.join(casePath_VSP_Results, "WaveDrag")
casePath_ituboom     = os.path.join(casePath, "ituboom")

casePath_machline = os.path.join(casePath, "MachLine-main")
casePath_dev      = os.path.join(casePath_machline, "dev_efe")
casePath_input    = os.path.join(casePath_dev, "Input_files")
casePath_meshes   = os.path.join(casePath_dev, "Meshes")
casePath_results  = os.path.join(casePath_dev, "Results")

# utility helpers
def _replaceLine(line_1, string_1, string_2):
    if string_1 in line_1:
        line_1 = line_1.replace(string_1, string_2)
    return line_1

def calculate_freestream(V_inf, aoa, beta):
    alpha_rad = np.radians(aoa)
    beta_rad = np.radians(beta)
    V_x = V_inf * np.cos(alpha_rad) * np.cos(beta_rad)
    V_y = V_inf * np.cos(alpha_rad) * np.sin(beta_rad)
    V_z = V_inf * np.sin(alpha_rad)
    return f"[{V_x}, {V_y}, {V_z}]"



# MachLine run function
def run_machline_analysis(stl_save_path, freestream, machNumber, Sref, formulation, casename="JWB"):
    input_template  = os.path.join(casePath, "base_input.json")
    temp_input_path = os.path.join(casePath, f"{casename}_input.json")
    final_input_path= os.path.join(casePath_input, f"{casename}_input.json")
    output_txt_path = os.path.join(casePath_results, f"{casename}_full_output.txt")
    coeff_txt_path  = os.path.join(casePath_results, f"{casename}_aerodynamic_coefficients.txt")

    # directory-check
    os.makedirs(casePath_input, exist_ok=True)
    os.makedirs(casePath_meshes, exist_ok=True)
    os.makedirs(casePath_results, exist_ok=True)

    with open(input_template) as fp:
        lines = fp.readlines()

    new_lines = []
    for line in lines:
        line = _replaceLine(line.strip(), "XXX_FLOWVELOCITY_XXX", freestream)
        line = _replaceLine(line, "XXX_MACHNUMBER_XXX", str(machNumber))
        stl_json_path = os.path.abspath(stl_save_path).replace("\\", "/")
        line = _replaceLine(line, "XXX_STLPATH_XXX", stl_json_path)
        line = _replaceLine(line, "XXX_REFAREA_XXX", str(Sref))
        line = _replaceLine(line, "XXX_FORMULATION_XXX", formulation)
        line = _replaceLine(line, "XXX_CASENAME_XXX", casename)
        new_lines.append(line)

    with open(temp_input_path, "w") as f:
        for line in new_lines:
            f.write(line + "\n")

    shutil.move(temp_input_path, final_input_path)

    # run MachLine
    os.chdir(casePath_machline)
    process = subprocess.Popen(
        [os.path.join(casePath_machline, "machline.exe"), final_input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    output_lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        output_lines.append(line)
    process.wait()

    # save output
    with open(output_txt_path, "w") as f:
        f.writelines(output_lines)

    # extract aero coefficients
    coeffs = {"Cx": 0.0, "Cy": 0.0, "Cz": 0.0, "CMx": 0.0, "CMy": 0.0, "CMz": 0.0}
    for line in output_lines:
        for key in coeffs:
            if f"{key}:" in line:
                try:
                    coeffs[key] = float(line.split(f"{key}:")[1].strip())
                except:
                    pass

    with open(coeff_txt_path, "w") as f:
        for key in coeffs:
            f.write(f"{coeffs[key]}\n")

    # copy .vtk to ituboom
    vtk_filename = f"{casename}.vtk"
    src = os.path.join(casePath_results, vtk_filename)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(casePath_ituboom, vtk_filename))

    return coeffs

# DV - design variables 
DV_ORDER = [
    "W_Tess_W", "W_Tess_U0", "W_Tess_U1", "W_Tess_U2", "W_sym",
    "W_xyzLoc_x","W_xyzLoc_y","W_xyzLoc_z",
    "W_xyzRot_x","W_xyzRot_y","W_xyzRot_z",
    "W_incidence",
    "W_root_chord_0","W_root_chord_1","W_root_chord_2",
    "W_tip_chord_0","W_tip_chord_1","W_tip_chord_2",
    "W_span_0","W_span_1","W_span_2",
    "W_sweep_0","W_sweep_1","W_sweep_2",
    "W_sweepLoc",
    "W_twist_0","W_twist_1","W_twist_2",
    "W_twistLoc",
    "W_dihedral_0","W_dihedral_1","W_dihedral_2",
    "F_Tess_W","F_SectTess_U",
    "F_xyzLoc_x","F_xyzLoc_y","F_xyzLoc_z",
    "F_xyzRot_x","F_xyzRot_y","F_xyzRot_z",
    "F_length",
    "F_xsec_1","F_xsec_2","F_xsec_3","F_xsec_4","F_xsec_5","F_xsec_6","F_xsec_7","F_xsec_8",
]
for sec in range(4):
    for surf in ("L","U"):
        for k in range(6):
            DV_ORDER.append(f"AF_{sec}_{surf}_{k}")

def get_default_dv():
    Nsec = 10
    xsec_default = np.linspace(0,1,Nsec).tolist()[1:-1]
    base_af = [
        ([-0.03209/2.67875, 0.00699/2.67875, -0.03478/2.67875, -0.01993/2.67875, -0.01560/2.67875, -0.02400/2.67875],
         [ 0.03209/2.67875, 0.07480/2.67875,  0.05004/2.67875,  0.07194/2.67875,  0.04672/2.67875,  0.04527/2.67875]),
        ([-0.03209, 0.00699, -0.03478, -0.01993, -0.01560, -0.02400],
         [ 0.03209, 0.07480,  0.05004,  0.07194,  0.04672,  0.04527]),
        ([-0.03272, 0.03713, -0.03756,  0.02647,  0.01877,  0.03600],
         [ 0.03272, 0.09817,  0.06234,  0.09869,  0.09521,  0.10005]),
        ([-0.02711, 0.02368, -0.01993,  0.01409,  0.02245,  0.03690],
         [ 0.02711, 0.11749,  0.02789,  0.13406,  0.07270,  0.10852]),
    ]
    dv = {
        "W_Tess_W":21,"W_Tess_U0":11,"W_Tess_U1":11,"W_Tess_U2":11,"W_sym":1,
        "W_xyzLoc_x":12.0,"W_xyzLoc_y":0.0,"W_xyzLoc_z":0.3,
        "W_xyzRot_x":0.0,"W_xyzRot_y":0.0,"W_xyzRot_z":0.0,
        "W_incidence":-0.9,
        "W_root_chord_0":21.43,"W_root_chord_1":8.043,"W_root_chord_2":4.389,
        "W_tip_chord_0":8.043,"W_tip_chord_1":4.389,"W_tip_chord_2":1.716,
        "W_span_0":1.264,"W_span_1":1.263,"W_span_2":2.051,
        "W_sweep_0":85.0,"W_sweep_1":75.0,"W_sweep_2":70.0,
        "W_sweepLoc":0.0,
        "W_twist_0":-0.05,"W_twist_1":-0.1,"W_twist_2":-0.1,
        "W_twistLoc":0.25,
        "W_dihedral_0":0.0,"W_dihedral_1":2.16,"W_dihedral_2":2.2,
        "F_Tess_W":37,"F_SectTess_U":15,
        "F_xyzLoc_x":0.0,"F_xyzLoc_y":0.0,"F_xyzLoc_z":0.0,
        "F_xyzRot_x":0.0,"F_xyzRot_y":0.0,"F_xyzRot_z":0.0,
        "F_length":20.0,
        "F_xsec_1":xsec_default[0],"F_xsec_2":xsec_default[1],"F_xsec_3":xsec_default[2],"F_xsec_4":xsec_default[3],
        "F_xsec_5":xsec_default[4],"F_xsec_6":xsec_default[5],"F_xsec_7":xsec_default[6],"F_xsec_8":xsec_default[7],
    }
    for sec,(L,U) in enumerate(base_af):
        for k in range(6):
            dv[f"AF_{sec}_L_{k}"]=float(L[k])
            dv[f"AF_{sec}_U_{k}"]=float(U[k])
    return dv

def _sorted_unit_interval(vals):
    v = np.clip(np.array(vals, dtype=float), 0.0, 1.0); v.sort()
    for i in range(1,len(v)):
        if v[i] <= v[i-1]:
            v[i] = min(1.0, v[i-1] + 1e-6)
    return v.tolist()

def dv_from_xVar(xVar):
    dv = get_default_dv()
    if isinstance(xVar, dict):
        dv.update(xVar); return dv
    if not isinstance(xVar, (list,tuple,np.ndarray)):
        return dv
    xVar = list(xVar)
    if len(xVar) == 12:
        dv["W_root_chord_1"]=xVar[0]; dv["W_root_chord_2"]=xVar[1]
        dv["W_tip_chord_1"]=xVar[2];  dv["W_tip_chord_2"]=xVar[3]
        dv["W_span_1"]=xVar[4];       dv["W_span_2"]=xVar[5]
        dv["W_sweep_1"]=xVar[6];      dv["W_sweep_2"]=xVar[7]
        dv["W_twist_1"]=xVar[8];      dv["W_twist_2"]=xVar[9]
        dv["W_dihedral_1"]=xVar[10];  dv["W_dihedral_2"]=xVar[11]
        return dv
    for i,key in enumerate(DV_ORDER):
        if i < len(xVar): dv[key] = xVar[i]
        else: break
    return dv

def objective_Function(xVar, casename="JWB"):
    dv = dv_from_xVar(xVar)
    VSP_Init()
    JWB = VSP_Geometry(casename=casename)
    stlPath = casePath + "/3.06_jwb_fuse_half.stl"
    mainWing_param = {
        'Tess_W': int(dv["W_Tess_W"]),
        'xyzLoc': [dv["W_xyzLoc_x"], dv["W_xyzLoc_y"], dv["W_xyzLoc_z"]],
        'xyzRot': [dv["W_xyzRot_x"], dv["W_xyzRot_y"], dv["W_xyzRot_z"]],
        'root_chords': [dv["W_root_chord_0"], dv["W_root_chord_1"], dv["W_root_chord_2"]],
        'tip_chords':  [dv["W_tip_chord_0"],  dv["W_tip_chord_1"],  dv["W_tip_chord_2"]],
        'spans':       [dv["W_span_0"],       dv["W_span_1"],       dv["W_span_2"]],
        'sweep':       [dv["W_sweep_0"],      dv["W_sweep_1"],      dv["W_sweep_2"]],
        'sweepLoc':    [dv["W_sweepLoc"]],
        'twist':       [dv["W_twist_0"],      dv["W_twist_1"],      dv["W_twist_2"]],
        'twistLoc':    dv["W_twistLoc"],
        'dihedral':    [dv["W_dihedral_0"],   dv["W_dihedral_1"],   dv["W_dihedral_2"]],
        'Tess_U': [int(dv["W_Tess_U0"]), int(dv["W_Tess_U1"]), int(dv["W_Tess_U2"])],
        'incidence': dv["W_incidence"],
    }
    mainWing = JWB.addWing(sym=bool(int(dv.get("W_sym", 1))))
    mainWing.getParams(mainWing_param)
    for sec_id in range(4):
        aLw = [dv[f"AF_{sec_id}_L_{k}"] for k in range(6)]
        aUp = [dv[f"AF_{sec_id}_U_{k}"] for k in range(6)]
        mainWing.appendAirfoil(aLw=aLw, aUp=aUp, sec_id=str(sec_id))
    Nsec = 10
    xsec_vals = _sorted_unit_interval([
        dv["F_xsec_1"], dv["F_xsec_2"], dv["F_xsec_3"], dv["F_xsec_4"],
        dv["F_xsec_5"], dv["F_xsec_6"], dv["F_xsec_7"], dv["F_xsec_8"],
    ])
    fuse1_param = {
        'Tess_W': int(dv["F_Tess_W"]),
        'xyzLoc': [dv["F_xyzLoc_x"], dv["F_xyzLoc_y"], dv["F_xyzLoc_z"]],
        'xyzRot': [dv["F_xyzRot_x"], dv["F_xyzRot_y"], dv["F_xyzRot_z"]],
        'length': float(dv["F_length"]),
        'SectTess_U': (int(dv["F_SectTess_U"]) * np.ones(Nsec - 1)).tolist(),
        'XSec_Loc': xsec_vals,
        'YSec_Loc': [],
        'ZSec_Loc': [],
    }
    fuse = JWB.addFuselage()
    fuse.params = fuse1_param
    fuse.fit_fuselage_ellipse(stlPath=stlPath, N_slice=Nsec, dp=7)
    aircraft = {'wing0': mainWing, 'fuselage': fuse}
    os.makedirs(casePath_geometries, exist_ok=True)
    os.makedirs(casePath_meshes, exist_ok=True)
    vsp_save_dir   = casePath_geometries
    stl_save_path  = os.path.join(casePath_meshes, f"{casename}.stl")
    vspCase = JWB.buildVSP(aircraft, casePath, casename)
    CompGeom_csv   = os.path.join(casePath, f"{casename}_CompGeom.csv")
    CompGeom_txt   = os.path.join(casePath, f"{casename}_CompGeom.txt")
    generated_vsp3 = os.path.join(casePath, f"{casename}.vsp3")
    generated_stl  = os.path.join(casePath, f"{casename}.stl")
    for src_file in [CompGeom_csv, CompGeom_txt, generated_vsp3]:
        if os.path.exists(src_file):
            shutil.move(src_file, os.path.join(vsp_save_dir, os.path.basename(src_file)))
    if os.path.exists(generated_stl):
        shutil.move(generated_stl, stl_save_path)
    return (vspCase, stl_save_path)


def run_full_case(machNumber_in=None, aoa_in=None, beta_in=None, rebuild_geometry=False, sample_idx=0):
    # inputs parameters
    machNumber_default  = 1.6
    aoa_default         = 3.06
    beta_default        = 0
    V_inf               = 500
    Sref                = 65.5
    formulation         = "dirichlet-morino"

    machNumber = machNumber_in if machNumber_in is not None else machNumber_default
    aoa        = aoa_in        if aoa_in        is not None else aoa_default
    beta       = beta_in       if beta_in       is not None else beta_default

    freestream = calculate_freestream(V_inf, aoa, beta)
    
    geometry_casename = "JWB"
    results_casename = f"JWB_{sample_idx}"

    stl_save_path = os.path.join(casePath_meshes, f"{geometry_casename}.stl")
    if rebuild_geometry:
        print(f"Building geometry '{geometry_casename}'...")
        xVar = [8.043, 4.389, 4.389, 1.716, 1.263, 2.051, 75, 70, -0.1, -0.1, 2.16, 2.2]
        objective_Function(xVar, casename=geometry_casename)
    else:
        print(f"Using existing geometry for '{geometry_casename}'...")
        if not os.path.exists(stl_save_path):
             raise FileNotFoundError(f"Expected geometry file not found at {stl_save_path}. Run sample 0 first.")

    # MachLine run
    coeffs = run_machline_analysis(
        stl_save_path=stl_save_path,
        freestream=freestream,
        machNumber=machNumber,
        Sref=Sref,
        formulation=formulation,
        casename=results_casename
    )
    print("Aero Coefficients:", coeffs)

    # read .vtk / run ituboom
    print(f"Running ITUBOOM example: {os.path.join(casePath_ituboom, 'example_1.py')}")
    result = subprocess.run(
        [sys.executable, "example_1.py", str(machNumber), results_casename, str(aoa)],
        cwd=casePath_ituboom,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    stdout_text = result.stdout
    print(stdout_text)

    # extract PLdB
    num = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    patterns = [rf'PLdB\s*[:=]?\s*{num}', rf'Perceived\s+Loudness\s*[:=]\s*{num}', rf'{num}\s*dB\b']
    pldb_val = None
    for pat in patterns:
        m = re.search(pat, stdout_text, flags=re.IGNORECASE)
        if m: pldb_val = float(m.group(1)); break
    if pldb_val is None:
        floats = re.findall(num, stdout_text)
        if floats: pldb_val = float(floats[-1])
    if pldb_val is None:
        raise RuntimeError("Could not parse PLdB from ITUBOOM output.\n" + stdout_text)
    
    # return results
    return coeffs, pldb_val

# main block to allow direct execution for testing
if __name__ == "__main__":
    coeffs, pldb = run_full_case()
    print("\nDirect Execution Summary")
    print(f"PLdB: {pldb}")
    print(f"Coefficients: {coeffs}")