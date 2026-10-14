"""Crimper kinematics as a class. Create one object per design."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from ipywidgets import interact, FloatSlider



def _rot(v, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    return np.array([ca*v[0] - sa*v[1], sa*v[0] + ca*v[1]])


def sat_clearance(A, B):
    """Separating-axis test for two convex polygons. >0: gap (mm), <0: overlap depth."""
    best = np.inf
    for poly in (A, B):
        for k in range(len(poly)):
            e = poly[(k+1) % len(poly)] - poly[k]
            ax = np.array([-e[1], e[0]]) / np.hypot(*e)
            pa, pb = A @ ax, B @ ax
            best = min(best, min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
    return -best


class Crimper:
    """
    N              number of crimp heads
    Rp             pivot circle radius (mm)
    bd, sd         drive pin: mm behind the tip, sideways offset (negative = inside the body)
    bp, sp         pivot pin: mm behind the tip, sideways offset
    width_factor   blade body width W = width_factor * Rp

    Treat the parameters as fixed; to change one, make a new object with .with_(...)
    """

    def __init__(self, N, Rp, bd, sd, bp, sp, pin_diameter, head_width,depth):

        self.N  = N
        self.Rp = Rp
        self.bd = bd
        self.sd = sd
        self.bp = bp
        self.sp = sp 
        self.W = head_width
        self.depth = depth
        self.pin_diameter = pin_diameter
        for name, m in self.pin_margins().items():
            worst = min(m.values())
            if worst < 0:
                raise ValueError(f"{name} pin (d={pin_diameter}) sticks out of the head by {-worst:.3f} mm "
                                 f"(margins to the edges: { {k: round(v, 3) for k, v in m.items()} })")
        if self.L >= Rp:
            raise ValueError("pivot-to-drive-pin distance L must be smaller than Rp")
        
        


    # ----- derived geometry -----

    @property
    def bl(self): return self.W / np.tan(2*np.pi/self.N)
    @property
    def Lb(self): return max(self.bp + 0.09*self.Rp, self.bl + 0.08*self.Rp)
    @property
    def a(self):  return self.bp - self.bd
    @property
    def c(self):  return self.sd - self.sp
    @property
    def L(self):  return float(np.hypot(self.a, self.c))

    @property
    def params(self):
        return dict(N=self.N, Rp=self.Rp, bd=self.bd, sd=self.sd, bp=self.bp, sp=self.sp,
                    width=self.W)


    def with_(self, **changes):
        """New crimper with some parameters changed, e.g. c.with_(bd=13)."""
        return Crimper(**{**self.params, **changes})


    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Crimper({p})"

    # ----- kinematics -----
    def blade_pose(self, delta_deg, blade_number=0):
        """Pose of one blade for a ring angle delta (0 = closed)."""
        d = np.deg2rad(delta_deg)
        S = self.L**2 - (self.Rp*np.sin(d))**2
        if S < 0:
            raise ValueError("ring angle beyond the kinematic limit asin(L/Rp)")
        r = self.Rp*np.cos(d) - np.sqrt(S)
        P = np.array([self.Rp, 0.0])
        Q = r*np.array([np.cos(d), np.sin(d)])
        w = Q - P
        phi = np.arctan2(w[1], w[0]) - np.arctan2(self.c, self.a)
        u = np.array([np.cos(phi), np.sin(phi)])
        n = np.array([-np.sin(phi), np.cos(phi)])
        T = Q + self.bd*u - self.sd*n
        assert np.allclose(T, P + self.bp*u - self.sp*n), "tip check failed"
        al = 2*np.pi*blade_number/self.N
        return dict(O=np.zeros(2), P=_rot(P, al), Q=_rot(Q, al), T=_rot(T, al),
                    u=_rot(u, al), n=_rot(n, al), phi=phi + al, r=r)


    def mechanism(self, drive_ring_angle, blade_number=0):
        p = self.blade_pose(drive_ring_angle, blade_number)
        return p["O"], p["Q"], p["P"], p["T"]


    def aperture(self, delta_deg):
        """Tip-circle diameter (mm)."""
        return 2*np.linalg.norm(self.blade_pose(delta_deg)["T"])


    def rate_of_aperture(self, delta_deg, blade_number =0):

        pose = self.blade_pose(delta_deg,blade_number)

        d   = np.deg2rad(delta_deg)
        S   = self.L**2 - (self.Rp*np.sin(d))**2

        r   = pose["r"]
        P   = pose["P"]
        Q   = pose["Q"]
        phi = pose["phi"]
        T   = pose["T"]
        

        w = Q - P
        A = T - Q
        

        r_per_delta = (self.Rp * r * np.sin(d))/np.sqrt(S)

        al = 2*np.pi*blade_number/self.N
        e  = np.array([np.cos(d + al), np.sin(d + al)])
        ep = np.array([-np.sin(d + al), np.cos(d + al)])

        Q_per_delta = r_per_delta * e + r * ep

        phi_per_delta = (w[0]*Q_per_delta[1] - w[1]*Q_per_delta[0])/ self.L**2

        u_per_delta = np.array([-np.sin(phi),np.cos(phi)])*phi_per_delta

        n_per_delta = np.array([np.cos(phi),np.sin(phi)])*(-1)*phi_per_delta

        T_per_delta = Q_per_delta + phi_per_delta * np.array([-A[1],A[0]])

        rho_per_delta = (T @ T_per_delta)/np.linalg.norm(T)      # mm per radian, same unit as the others

        diameter_per_delta = 2*rho_per_delta                      # mm per radian

        diameter_per_deg = diameter_per_delta*np.pi/180           # separate key, mm per degree, for reading only

        rate = dict(
            delta_r=r_per_delta,
            delta_Q= Q_per_delta,
            delta_phi = phi_per_delta,
            delta_u = u_per_delta,
            delta_n = n_per_delta,
            drho = rho_per_delta,
            diameter_per_delta = diameter_per_delta,
            diameter_per_deg = diameter_per_deg
            )
        
        return rate


    #------  Torque and Forces ---------

    def ring_torque_curve(self, F, efficiency=0.6, per_blade=True, n_points=300):
        """
        tau_ring = F_total * |d rho / d delta| / efficiency      (virtual work)
        F         : radial force of one tip in N (number), or a function F(D_mm) -> N
        per_blade : True: F is for ONE blade (multiplied by N); False: F is the total force
        returns angles (deg), apertures (mm), torque (N*m)
        """
        lim = self.limits()
        angles = np.linspace(lim["collision_free_min"], lim["collision_free_max"], n_points)
        D = np.array([self.aperture(a) for a in angles])
        drho = np.array([abs(self.rate_of_aperture(a)["drho"]) for a in angles])   # mm per rad
        F_each = np.array([F(x) for x in D], dtype=float) if callable(F) else np.full(D.shape, float(F))
        F_tot = F_each*self.N if per_blade else F_each
        return angles, D, F_tot*drho/1000.0/efficiency          # N*mm -> N*m


    def ring_torque(self, F, efficiency=0.6, per_blade=True, n_points=300):
        """Worst-case ring torque over the collision-free range and where it occurs."""
        angles, D, tau = self.ring_torque_curve(F, efficiency, per_blade, n_points)
        i = int(np.argmax(tau))
        return dict(torque_Nm=tau[i], at_angle_deg=angles[i], at_aperture_mm=D[i])


    def motor_torque(self, F, gear_ratio, efficiency=0.6, gear_efficiency=0.9, **kw):
        """Motor torque for a reduction of gear_ratio (motor turns per ring turn)."""
        return self.ring_torque(F, efficiency, **kw)["torque_Nm"]/(gear_ratio*gear_efficiency)


    def pin_forces(self, delta_deg, F):
        """
        Forces on blade 0 for a stent force F (N) on ONE blade, acting on the flat face.
        Frictionless radial slot. Returns F_slot (on the drive pin), R_pivot (on the pivot pin).
        Components are in the world frame, in N.
        """
        cross = lambda p, q: p[0]*q[1] - p[1]*q[0]
        d = np.deg2rad(delta_deg)
        pose = self.blade_pose(delta_deg, 0)
        P, Q, T, u, n = pose["P"], pose["Q"], pose["T"], pose["u"], pose["n"]

        H  = T - (T @ u)*u                       # contact point on the flat face
        m  = np.sign(T @ n)*n                    # normal pointing away from O
        Fs = F*m                                 # stent force on the blade
        ep = np.array([-np.sin(d), np.cos(d)])   # direction of the slot force (perpendicular to the slot)

        Ft      = -cross(H - P, Fs)/cross(Q - P, ep)   # moment balance about P
        F_slot  = Ft*ep
        R_pivot = -(Fs + F_slot)                       # force balance
        return dict(F_t=Ft, F_slot=F_slot, R_pivot=R_pivot,
                    slot_N=abs(Ft), pivot_N=float(np.linalg.norm(R_pivot)),
                    r=pose["r"], contact_point=H,
                    ring_torque_per_blade_Nm=abs(Ft)*pose["r"]/1000.0)


    def max_pin_forces(self, F, n_points=300, per_blade=True):
        """
        Largest pin loads over the collision-free ring range (Model A, frictionless slot).
        F         : stent force on ONE blade in N (number), or a function F(D_mm) -> N
        per_blade : False if F is the TOTAL stent force over all blades (it is divided by N)
        Returns the maxima, where they occur, and the full curves for plotting.
        """

        lim = self.limits()
        angles = np.linspace(lim["collision_free_min"], lim["collision_free_max"], n_points)
        D = np.array([self.aperture(a) for a in angles])
        F_each = np.array([F(x) for x in D], dtype=float) if callable(F) else np.full(D.shape, float(F))
        if not per_blade:
            F_each = F_each/self.N
        rows = [self.pin_forces(a, f) for a, f in zip(angles, F_each)]
        slot = np.array([r["slot_N"] for r in rows])
        piv  = np.array([r["pivot_N"] for r in rows])
        i, j = int(np.argmax(slot)), int(np.argmax(piv))
        return dict(max_slot_N=slot[i], slot_at_angle_deg=angles[i], slot_at_aperture_mm=D[i],
                    max_pivot_N=piv[j], pivot_at_angle_deg=angles[j], pivot_at_aperture_mm=D[j],
                    max_pivot_vector_N=rows[j]["R_pivot"],
                    angles=angles, aperture=D, slot_N=slot, pivot_N=piv, F_each=F_each)

    #------ Construction and design ------

    def head_outline(self):
        """Head outline in blade coordinates (b along the blade from the tip, s across it)."""
        return np.array([[0, 0], [self.bl, -self.W], [self.Lb, -self.W], [self.Lb, 0]], float)

    def _edge_distances(self, which):
        """Signed distance (mm) from the pin centre to each edge line; positive = inside the head.
        The outline is convex, so the smallest value is the true distance to the boundary."""
        p = np.array((self.bd, self.sd) if which == "drive" else (self.bp, self.sp), float)
        pts = self.head_outline()
        names = ["angled_face", "far_edge", "back_end", "flat_face"]
        out = {}
        for i, nm in enumerate(names):
            a, b = pts[i], pts[(i + 1) % 4]
            ab = b - a
            out[nm] = float((ab[0]*(p[1]-a[1]) - ab[1]*(p[0]-a[0])) / np.hypot(*ab))
        return out

    def pin_margins(self):
        """Material left between each pin and every head edge (mm). Negative = sticks out."""
        r = self.pin_diameter/2
        return {w: {k: v - r for k, v in self._edge_distances(w).items()} for w in ("drive", "pivot")}

    def max_pin_for_width(self, which="drive", edge_margin_mm=1.0):
        """Largest pin diameter that fits in the head, keeping edge_margin_mm to the nearest edge."""
        return 2*(min(self._edge_distances(which).values()) - edge_margin_mm)

    def min_pin_diameter(self, force_N, sy_MPa, safety_factor=2.0, shear_planes=2,
                         bearing_length_mm=None, span_mm=None,
                         support="simple", arm_mm=None, bearing_allow_MPa=None):
        """
        Minimum diameter (mm) for ONE pin carrying force_N. Uses only pin geometry, not self.depth.

        bearing_length_mm : length of pin inside the part that loads it (bearing check, and the length
                            over which the load is spread). None -> bearing check skipped.
        span_mm           : distance between the pin's two supports (simple) -> bending check.
                            None -> bending check skipped (unless support="cantilever" with arm_mm).
        Shear is always checked.
        """
        sig, tau = sy_MPa/safety_factor, 0.577*sy_MPa/safety_factor
        res = {"shear": np.sqrt(4*force_N/(np.pi*shear_planes*tau))}
        M = None
        if support == "simple" and span_mm is not None:
            lb = 0.0 if bearing_length_mm is None else bearing_length_mm   # lb=0 -> point load, conservative
            M = force_N/4*(span_mm - lb/2)
        elif support == "cantilever" and arm_mm is not None:
            M = force_N*arm_mm
        if M is not None:
            res["bending"] = (32*M/(np.pi*sig))**(1/3)
        if bearing_length_mm is not None:
            allow = sig if bearing_allow_MPa is None else bearing_allow_MPa
            res["bearing"] = force_N/(bearing_length_mm*allow)
        gov = max(res, key=res.get)
        return dict(d_min=res[gov], governed_by=gov, checks=res, M_Nmm=M,
                    skipped=[k for k in ("bending", "bearing") if k not in res])

    def check_pins(self, F, sy_MPa, safety_factor=2.0, edge_margin_mm=1.0, **kw):
        """Compare the pins you chose with the loads: strength and fit, for both pins."""
        r = self.max_pin_forces(F)
        mg = self.pin_margins()
        out = {}
        for name, load in (("drive", r["max_slot_N"]), ("pivot", r["max_pivot_N"])):
            m = self.min_pin_diameter(load, sy_MPa, safety_factor, **kw)
            out[name] = dict(load_N=load, d_min=m["d_min"], governed_by=m["governed_by"],
                             d_chosen=self.pin_diameter, strength_ok=self.pin_diameter >= m["d_min"],
                             margins_mm=mg[name], fit_ok=min(mg[name].values()) >= edge_margin_mm,
                             d_max_by_width=self.max_pin_for_width(name, edge_margin_mm))
        return out

    
    # ----- blade outline and gap -----

    def polygon(self, pose):
        local = np.array([[0, 0], [self.bl, -self.W], [self.Lb, -self.W], [self.Lb, 0]])
        return np.array([pose["T"] - b*pose["u"] + s*pose["n"] for b, s in local])


    def gap(self, delta_deg):
        """Clearance between neighbouring blades (mm; negative = overlap)."""
        return sat_clearance(self.polygon(self.blade_pose(delta_deg, 0)),
                             self.polygon(self.blade_pose(delta_deg, 1)))


    def gap_formula(self, delta_deg):
        """Same gap from d = 2*rho*sin(pi/N)*cos(omega + pi/N)."""
        p = self.blade_pose(delta_deg)
        T = p["T"]; rho = np.linalg.norm(T)
        omega = (p["phi"] - (np.arctan2(T[1], T[0]) + np.pi) + np.pi) % (2*np.pi) - np.pi
        return 2*rho*np.sin(np.pi/self.N)*np.cos(omega + np.pi/self.N)

    # ----- drive angle limits -----

    def limits(self, tol=1e-6, margin_deg=2.0, step_deg=0.01):
        kin = np.degrees(np.arcsin(self.L/self.Rp))
        top = kin - margin_deg
        ds = np.arange(0.0, top + step_deg, step_deg)
        ok = np.array([self.gap(x) >= -tol for x in ds])
        first = last = None
        if ok.any():
            i0 = int(np.argmax(ok)); first = ds[i0]
            bad = np.where(~ok[i0:])[0]
            last = ds[i0 + bad[0] - 1] if len(bad) else ds[-1]
        return dict(kinematic_max=kin, usable_max=top, min=0.0,
                    collision_free_min=first, collision_free_max=last,
                    aperture_at_min=self.aperture(first) if first is not None else None,
                    aperture_at_max=self.aperture(last) if last is not None else None)

    #--------Vizu-------------

    def inscribed_aperture(self, delta_deg):
        """Diameter of the circle that fits between the flat faces (what the stent sees)."""
        p = self.blade_pose(delta_deg)
        T, u = p["T"], p["u"]
        return 2*abs(T[0]*u[1] - T[1]*u[0])


    def plot(self, delta_deg, ax=None, tol=1e-6, show_slots=True, show_circles=True,
             show_pins=True, labels=False, view=None):
        """
        Draw the crimper at ring angle delta_deg.
        pin_d   : drawn pin diameter and slot width (mm)
        tol     : blades are drawn red when the gap is below -tol (mm)
        labels  : annotate O, P, Q and T on blade 0
        view    : half-width of the drawing window in mm (default 1.2*Rp)
        """
        
        created = ax is None
        if created:
            _, ax = plt.subplots(figsize=(7, 7))
        gap = self.gap(delta_deg)
        hit = gap < -tol
        d_top = np.radians(np.degrees(np.arcsin(self.L/self.Rp)) - 2.0)
        r_lo = self.Rp - self.L
        r_hi = self.Rp*np.cos(d_top) - np.sqrt(self.L**2 - (self.Rp*np.sin(d_top))**2)
        for i in range(self.N):
            pose = self.blade_pose(delta_deg, i)
            if show_slots:                                  # radial hub slot for blade i
                ang = 2*np.pi*i/self.N + np.radians(delta_deg)
                e, nrm = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
                h = self.pin_diameter
                ax.add_patch(Polygon([r_lo*e - h*nrm, r_hi*e - h*nrm, r_hi*e + h*nrm, r_lo*e + h*nrm],
                                     closed=True, facecolor="#EF9F27", alpha=0.35, edgecolor="none", zorder=2))
            face = "#F4A6A6" if hit else ("#9FC5E8" if i == 0 else "#D9E6F2")
            ax.add_patch(Polygon(self.polygon(pose), closed=True, facecolor=face,
                                 edgecolor="#2B4C6F", lw=1.2 if i == 0 else 0.8))
            if show_pins:
                ax.add_patch(Circle(pose["P"], self.pin_diameter/2, color="#7F77DD", zorder=3))
                ax.add_patch(Circle(pose["Q"], self.pin_diameter/2, color="#EF9F27", zorder=3))
        p0 = self.blade_pose(delta_deg, 0)
        D, Dins = self.aperture(delta_deg), self.inscribed_aperture(delta_deg)
        if show_circles:
            ax.add_patch(Circle((0, 0), self.Rp, fill=False, ls=":", ec="gray", lw=0.8))
            if D > 1e-6:
                ax.add_patch(Circle((0, 0), D/2, fill=False, ls="--", ec="#1D9E75", lw=1.4))
                ax.add_patch(Circle((0, 0), Dins/2, fill=False, ls=":", ec="#D85A30", lw=1.4))
        if labels:
            for name, pt, col in (("O", p0["O"], "k"), ("P", p0["P"], "#7F77DD"),
                                  ("Q", p0["Q"], "#EF9F27"), ("T", p0["T"], "k")):
                ax.annotate(name, pt, xytext=(5, 5), textcoords="offset points", color=col, weight="bold")
            ax.plot(*p0["O"], "k.", ms=5); ax.plot(*p0["T"], "k.", ms=5)
        half = view or 1.2*self.Rp
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_title(f"ring angle {delta_deg:.1f}°   D = {D:.2f} mm (inscribed {Dins:.2f})   "
                     f"gap = {gap:+.3f} mm" + ("  COLLISION" if hit else ""), fontsize=10)
        if created:
            plt.show()
        return ax


    def plot_sequence(self, angles=None, ncols=3, **kwargs):
        """Several ring angles side by side. Default: closed, 1/3, 2/3 and the collision-free maximum."""
        if angles is None:
            top = self.limits()["collision_free_max"]
            angles = [0.0, top/3, 2*top/3, top]
        nrows = int(np.ceil(len(angles)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
        for ax in axes.flat[len(angles):]:
            ax.axis("off")
        for ax, ang in zip(axes.flat, angles):
            self.plot(ang, ax=ax, **kwargs)
        fig.tight_layout()
        plt.show()
        return fig


    def interactive(self, **kwargs):
        """Slider for the ring angle (needs ipywidgets in Jupyter)."""
        lim = self.limits()
        top = lim["collision_free_max"]
        bot = lim["collision_free_min"]
        def draw(delta):
            self.plot(delta, **kwargs)
        return interact(draw, delta=FloatSlider(min=bot, max=top, step=0.5, value=min(20.0, top),
                                                description="ring angle"))


    def plot_gap_over_angle(self):

        plt.xlabel("Drive angle [deg]")
        plt.ylabel("Gap [mm]")
        plt.title("Gap over Angle")


        lim = self.limits()
        top = lim["collision_free_max"]
        bot = lim["collision_free_min"]
        angle_list = np.linspace(bot, top, num=1000)

        gap_at_angle =[]
        for angle in angle_list:
            gap_at_angle.append(self.gap(angle))

        plt.plot(angle_list,gap_at_angle)
        plt.show()


    def plot_rate_of_aperture(self):
        plt.xlabel("Drive angle [deg]")
        plt.ylabel("diameter/angle [mm/]")
        plt.title("change rate of diameter per angle")

        lim = self.limits()
        top = lim["collision_free_max"]
        bot = lim["collision_free_min"]
        angle_list = np.linspace(bot, top, num=1000)

        delta_d =[]
        for angle in angle_list:
            delta_d.append(self.rate_of_aperture(angle)["diameter_per_delta"])

        print(delta_d)
        #plt.plot(angle_list, dydx)
        plt.plot(angle_list, delta_d)
        plt.show()


    def plot_diameter_by_angle(self):
        plt.xlabel("Drive angle [deg]")
        plt.ylabel("Diameter [mm]")
        plt.title("Diameter over actuation angle")

        lim = self.limits()
        top = lim["collision_free_max"]
        bot = lim["collision_free_min"]
        angle_list = np.linspace(bot, top, num=1000)
        
        diameter =[]
        for angle in angle_list:
            diameter.append(self.aperture(angle))

        plt.plot(angle_list,diameter)
        plt.show()