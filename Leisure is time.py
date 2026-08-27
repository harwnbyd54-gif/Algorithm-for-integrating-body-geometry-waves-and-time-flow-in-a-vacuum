import numpy as np
from scipy.spatial import ConvexHull

class SpacetimeVacuumSimulator:
    """
    محاكي تكامل هندسة الجسم، الموجات، وتدفق الزمن في الفراغ.
    يعتمد على النسبية العامة الضعيفة (Weak Field Approximation).
    
    المعادلات الأساسية:
    -------------------
    1. ∇²Φ ≈ 4πGρ          (معادلة بواسون للجاذبية الضعيفة)
    2. dτ/dt = √(1 + 2Φ/c²)  (التمدد الزمني الضعيف)
    3. a = -∇Φ              (تسارع الجاذبية)
    4. dx/dτ = v, dv/dτ = a  (الحركة في الزمن الخاص)
    """
    
    def __init__(self, G=6.67430e-11, c=299792458.0):
        self.G = G          # ثابت الجاذبية العامة [m³ kg⁻¹ s⁻²]
        self.c = c          # سرعة الضوء [m/s]
        self.c2 = c ** 2    # c² للاستخدام المتكرر
    
    # ------------------------------------------------------------------
    # 1. حساب الكتلة الحقيقية لكل رأس
    # ------------------------------------------------------------------
    def compute_vertex_masses(self, vertices, faces, material_density):
        """
        حساب كتلة كل رأس باستخدام حجم الخلية المجاورة (Voronoi-like approximation).
        
        Parameters:
        -----------
        vertices : (N, 3) array
            إحداثيات الرؤوس في الفضاء 3D.
        faces : (M, 3) array
            مؤشرات المثلثات (triangular mesh).
        material_density : float or (N,) array
            كثافة المادة [kg/m³].
        
        Returns:
        --------
        mass : (N,) array
            كتلة كل رأس [kg].
        volumes : (N,) array
            حجم الخلية المجاورة لكل رأس [m³].
        """
        N = len(vertices)
        if np.isscalar(material_density):
            material_density = np.full(N, material_density)
        
        # تقريب الحجم: مساحة الأوجه المجاورة × عمق تقريبي
        vertex_areas = np.zeros(N)
        if faces is not None:
            for face in faces:
                v0, v1, v2 = vertices[face]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                vertex_areas[face] += area / 3.0
            
            # تقدير العمق باستخدام ConvexHull
            try:
                hull = ConvexHull(vertices)
                total_vol = hull.volume
                total_area = vertex_areas.sum()
                depth = total_vol / (total_area + 1e-12)
            except Exception:
                depth = 1e-3  # قيمة افتراضية
        else:
            vertex_areas = np.full(N, 1.0)
            depth = 1e-3
        
        volumes = vertex_areas * depth
        return volumes * material_density, volumes
    
    # ------------------------------------------------------------------
    # 2. حساب تركيز طاقة الموجة عند الزوايا الحادة
    # ------------------------------------------------------------------
    def compute_wave_geometric_resonance(self, vertices, faces, wave):
        """
        حساب تركيز طاقة الموجة عند الزوايا الحادة (Sharp Angles).
        
        الفكرة الفيزيائية:
        --------------------
        الزوايا الحادة (dihedral angles قريبة من 90°) تعمل كـ "هوائيات" هندسية
        تركز طاقة الموجة في الفراغ. هذا مستوحى من البصريات/الموجات الكهرومغناطيسية
        حيث تتركز الحقول عند النقاط الحادة.
        
        Parameters:
        -----------
        vertices : (N, 3) array
        faces : (M, 3) array
        wave : dict
            {'amplitude': float, 'frequency': float}
        
        Returns:
        --------
        resonance : (N,) array
            طاقة الموجة المركزة عند كل رأس [J].
        """
        N = len(vertices)
        resonance = np.zeros(N)
        if faces is None:
            return resonance
        
        amp = wave.get('amplitude', 1.0)
        freq = wave.get('frequency', 1.0)
        
        # خريطة الأوجه المجاورة لكل رأس
        adj_faces = [[] for _ in range(N)]
        for idx, face in enumerate(faces):
            for v in face:
                adj_faces[v].append(idx)
        
        # حساب الأنظمة (Normals) لكل وجه
        normals = np.zeros((len(faces), 3))
        for idx, face in enumerate(faces):
            v0, v1, v2 = vertices[face]
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 0:
                normals[idx] = n / norm
        
        # حساب الحدة (Sharpness) عند كل رأس
        for v in range(N):
            af = adj_faces[v]
            if len(af) < 2:
                continue
            sharpness = 0.0
            for i in range(len(af)):
                for j in range(i + 1, len(af)):
                    cos_a = np.clip(np.dot(normals[af[i]], normals[af[j]]), -1.0, 1.0)
                    angle = np.arccos(abs(cos_a))
                    # الحدة تتناسب مع sin(angle):
                    # 0 للسطح المستوي، 1 للزاوية القائمة (أشد حدة)
                    sharpness += np.sin(angle)
            resonance[v] = amp * freq * sharpness
        
        return resonance
    
    # ------------------------------------------------------------------
    # 3. الجهد الجاذبي (N-body مع Softening)
    # ------------------------------------------------------------------
    def gravitational_potential(self, vertices, mass, epsilon=0.1):
        """
        حساب الجهد الجاذبي عند كل رأس باستخدام تقريب N-body مع softening.
        
        Φ_i = -G * Σ_j (m_j / sqrt(|r_ij|² + ε²))
        
        هذا تقريب عددي لمعادلة بواسون: ∇²Φ = 4πGρ
        
        Parameters:
        -----------
        vertices : (N, 3) array
        mass : (N,) array
        epsilon : float
            معامل softening لتجنب الشوشرة العددية عند r→0.
        
        Returns:
        --------
        phi : (N,) array
            الجهد الجاذبي [J/kg] أو [m²/s²].
        """
        N = len(vertices)
        phi = np.zeros(N)
        for i in range(N):
            dr = vertices - vertices[i]  # (N, 3)
            r2 = np.sum(dr**2, axis=1) + epsilon**2
            r2[i] = np.inf  # تجنب الذات (self-interaction)
            phi[i] = -self.G * np.sum(mass / np.sqrt(r2))
        return phi
    
    # ------------------------------------------------------------------
    # 4. التدرج التحليلي للجهد
    # ------------------------------------------------------------------
    def gravitational_gradient(self, vertices, mass, epsilon=0.1):
        """
        حساب التدرج التحليلي للجهد الجاذبي.
        
        ∇_i Φ = G * Σ_j m_j * (r_i - r_j) / (|r_ij|² + ε²)^(3/2)
        
        Returns:
        --------
        grad : (N, 3) array
            متجه التدرج عند كل رأس [m/s²].
        """
        N = len(vertices)
        grad = np.zeros((N, 3))
        for i in range(N):
            dr = vertices[i] - vertices  # (N, 3)
            r2 = np.sum(dr**2, axis=1) + epsilon**2
            r2[i] = np.inf
            grad[i] = self.G * np.sum(
                (mass[:, None] * dr) / (r2[:, None] ** 1.5), axis=0
            )
        return grad
    
    # ------------------------------------------------------------------
    # 5. عامل التمدد الزمني (Time Dilation)
    # ------------------------------------------------------------------
    def time_dilation_factor(self, phi):
        """
        حساب عامل التمدد الزمني في النسبية العامة الضعيفة.
        
        dτ/dt = √(1 + 2Φ/c²)
        
        حيث:
        - dτ : الزمن الخاص (Proper Time) عند الرأس.
        - dt : الزمن المنسوب (Coordinate Time) للناظر البعيد.
        - Φ  : الجهد الجاذبي (سالب للأجسام العادية).
        
        الحماية العددية:
        ------------------
        إذا تجاوز |Φ| قيمة c²/2 (قرب الأفق الحدث للثقب الأسود)،
        نثبت القيمة عند حد آمن لتجنب الأعداد التخيلية.
        """
        arg = 1.0 + 2.0 * phi / self.c2
        return np.sqrt(np.maximum(arg, 1e-12))
    
    # ------------------------------------------------------------------
    # 6. المحاكاة الرئيسية
    # ------------------------------------------------------------------
    def simulate(self, geometry_mesh, faces, material_density, wave, time_steps, dt):
        """
        تشغيل المحاكاة الزمنية.
        
        Parameters:
        -----------
        geometry_mesh : (N, 3) array
            إحداثيات الرؤوس الأولية.
        faces : (M, 3) array or None
            مؤشرات المثلثات.
        material_density : float or (N,) array
            كثافة المادة [kg/m³].
        wave : dict
            {'amplitude': float [J], 'frequency': float [Hz]}
        time_steps : int
            عدد خطوات الزمن.
        dt : float
            فترة الزمن المنسوب (Coordinate Time) [s].
        
        Returns:
        --------
        final_vertices : (N, 3) array
            الإحداثيات النهائية بعد المحاكاة.
        dilation_history : (time_steps, N) array
            تاريخ عامل التمدد الزمني لكل رأس في كل خطوة.
        """
        vertices = geometry_mesh.copy().astype(float)
        N = len(vertices)
        velocities = np.zeros((N, 3))
        
        # 1. حساب الكتلة والحجم
        mass, vol = self.compute_vertex_masses(vertices, faces, material_density)
        
        # 2. طاقة الموجة الهندسية (تركيز عند الزوايا الحادة)
        wave_energy = self.compute_wave_geometric_resonance(vertices, faces, wave)
        
        # 3. الكتلة الفعالة (كتلة + طاقة الموجة/c²)
        # في النسبية: T_00 = ρc². نضيف الموجة كمساهمة في الطاقة.
        effective_mass = mass + wave_energy / self.c2
        
        dilation_history = []
        
        for step in range(time_steps):
            # 4. الجهد الجاذبي
            phi = self.gravitational_potential(vertices, effective_mass)
            
            # 5. عامل التمدد الزمني
            dilation = self.time_dilation_factor(phi)
            dilation_history.append(dilation.copy())
            
            # 6. الزمن الخاص لكل نقطة (varies per vertex)
            d_tau = dt * dilation
            
            # 7. تسارع الجاذبية
            grad_phi = self.gravitational_gradient(vertices, effective_mass)
            accel = -grad_phi  # a = -∇Φ
            
            # 8. تحديث الحركة (Euler-Cromer للاستقرار العددي)
            velocities += accel * d_tau[:, None]
            vertices += velocities * d_tau[:, None]
        
        return vertices, np.array(dilation_history)


# ===================================================================
# مثال على الاستخدام
# ===================================================================
if __name__ == "__main__":
    # هندسة مكعب وحدة (1×1×1 متر)
    verts = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=float)
    
    faces = np.array([
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [2,3,7], [2,7,6],
        [0,3,7], [0,7,4], [1,2,6], [1,6,5]
    ])
    
    sim = SpacetimeVacuumSimulator()
    
    final_verts, dilation = sim.simulate(
        geometry_mesh=verts,
        faces=faces,
        material_density=7850.0,       # كثافة الحديد [kg/m³]
        wave={'amplitude': 1e-6, 'frequency': 1e12},  # موجة ترددية
        time_steps=50,
        dt=1e-6                        # 1 μs per step
    )
    
    print("=" * 50)
    print("نتائج المحاكاة")
    print("=" * 50)
    print(f"الزمن المنسوب الكلي: {50 * 1e-6:.2e} s")
    print(f"متوسط التمدد الزمني (آخر خطوة): {dilation[-1].mean():.15f}")
    print(f"الحد الأدنى (أبطأ زمن): {dilation[-1].min():.15f}")
    print(f"الحد الأقصى (أسرع زمن): {dilation[-1].max():.15f}")
    print(f"الإزاحة المتوسطة للرؤوس: {np.linalg.norm(final_verts - verts, axis=1).mean():.6e} m")
