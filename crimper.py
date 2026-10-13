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

    def __init__(self, N, Rp, bd, sd, bp, sp, pin_diameter, head_width):
        self.N  = N
        self.Rp = Rp
        self.bd = bd
        self.sd = sd
        self.bp = bp
        self.sp = sp 
        self.W = head_width
        if not (-self.W     <= sd-(pin_diameter/2) and
                0           >= sd+(pin_diameter/2) and
                -self.W    <= sp-(pin_diameter/2) and
                0           >= sp-(pin_diameter/2)):
            warnings.warn("a pin is is outside the blade body: for a flat + angled blade "
                          "sd and sp should satisfy -W <= s <= 0")
    
            raise ValueError("pin outside the crimp head body choose an appropiate postion for the pins")
        
        if self.L >= Rp:
            warnings.warn("pivot-to-drive-pin distance L must be smaller than Rp")

        self.pin_diameter = pin_diameter


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

        d = np.deg2rad(delta_deg)
        S = self.L**2 - (self.Rp*np.sin(d))**2
        if S < 0:
            raise ValueError("ring angle beyond the kinematic limit asin(L/Rp)")
        r = self.Rp*np.cos(d) - np.sqrt(S)

        P = np.array([self.Rp, 0.0])
        Q = r*np.array([np.cos(d), np.sin(d)])
        w = Q - P
        phi = np.arctan2(w[1], w[0]) - np.arctan2(self.c, self.a)

        r_per_delta = (self.Rp * r * np.sin(d))/np.sqrt(S)

        Q_per_delta = r_per_delta * np.array([np.cos(d), np.sin(d)]) + r * np.array([-np.sin(d), np.cos(d)])

        phi_per_delta = (Q_per_delta[1]*w[0] - Q_per_delta[0]*w[1])/ self.L**2

        u_per_delta = np.array([-np.sin(phi),np.cos(phi)])*phi_per_delta

        n_per_delta = np.array([np.cos(phi),np.sin(phi)])*(-1)*phi_per_delta

        rate_of_aperture = 2*(Q_per_delta + (self.bd*u_per_delta) - (self.sd*n_per_delta ))

        rate = dict(
            delta_r=r_per_delta,
            delta_Q= Q_per_delta,
            delta_phi = phi_per_delta,
            delta_u = u_per_delta,
            delta_n = n_per_delta,
            rate_of_aperture = rate_of_aperture
            )
        return rate

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
        top = self.limits()["collision_free_max"]
        def draw(delta):
            self.plot(delta, **kwargs)
        return interact(draw, delta=FloatSlider(min=0.0, max=top, step=0.5, value=min(20.0, top),
                                                description="ring angle"))

    def plot_gap_over_angle(self):

        plt.xlabel("Drive angle [deg]")
        plt.ylabel("Gap [mm]")
        plt.title("Gap over Angle")
        max_no_colision = self.limits()["collision_free_max"]

        angle_list = np.linspace(0.0, max_no_colision, num=1000)
        gap_at_angle =[]
        for angle in angle_list:
            gap_at_angle.append(self.gap(angle))

        plt.plot(angle_list,gap_at_angle)
        plt.show()