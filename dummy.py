import os
os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import keras, h5py, glob, json, numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from scipy.special import genlaguerre
lay = keras.layers


USE_H5_DATA   = True           # False → use synthetic generate_dataset()
H5_FOLDER     = "/home/hpc/b129dc/b129dc30/configA_data"
N_RES         = 64             
ORDER         = 2


def lg_mode_indices(max_order=4):
    modes = []
    for p in range(max_order + 1):
        for l in range(-max_order, max_order + 1):
            
            if p + abs(l) <= max_order: 
                modes.append((p, l))
    return modes

MODES   = lg_mode_indices(ORDER)
N_COEFF = len(MODES)
print(f"Modes: {N_COEFF}", MODES)

def make_grid(N=32):
    x = np.linspace(-2, 2, N); y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)
    return np.sqrt(X**2+Y**2), np.arctan2(Y, X)

def lg_mode(p, l, r, theta):
    l_abs = abs(l)
    L   = genlaguerre(p, l_abs)(2*r**2)
    amp = ((r * np.sqrt(2))**l_abs) * np.exp(-(r**2)) * L
    mode = amp * np.exp(1j*l*theta)
    return mode / np.sqrt(np.sum(np.abs(mode)**2))

def synthesize_phase(coeffs, n_res=32):
    r, theta = make_grid(n_res)
    E = np.zeros_like(r, dtype=np.complex128)
    for c, (p, l) in zip(coeffs, MODES):
        E += c * lg_mode(p, l, r, theta)
    return np.angle(E)

def generate_dataset(N_samples=5000, n_res=32):
    coeffs = np.random.randn(N_samples, N_COEFF) + 1j*np.random.randn(N_samples, N_COEFF)
    coeffs /= np.sqrt(np.sum(np.abs(coeffs)**2, axis=1, keepdims=True))
    Y = np.zeros((N_samples, 2*N_COEFF), dtype=np.float32)
    Y[:, 0::2] = coeffs.real;  Y[:, 1::2] = coeffs.imag
    X = np.zeros((N_samples, n_res, n_res, 1), dtype=np.float32)
    for i in range(N_samples):
        X[i, ..., 0] = synthesize_phase(coeffs[i], n_res)
    return X, Y

def load_h5_dataset(folder_path, modes_list):
    mode_to_idx = {f"p{p}l{l}": i for i, (p, l) in enumerate(modes_list)}
    X_list, Y_list = [], []
    for fp in glob.glob(os.path.join(folder_path, "*.h5")):
        with h5py.File(fp, 'r') as f:
            X_list.append(f['phase'][:][..., np.newaxis])
            coeffs_c = np.zeros(len(modes_list), dtype=complex)
            mixing   = json.loads(f.attrs.get('mixing_coefficients_json', '{}'))
            for md in mixing.values():
                if md['mode'] in mode_to_idx:
                    idx = mode_to_idx[md['mode']]
                    #coeffs_c[idx] = md['amplitude'] * np.exp(1j*md['phase_rad'])
                    coeffs_c[idx] += md['amplitude'] * np.exp(1j * md['phase_rad'])
            norm = np.sqrt(np.sum(np.abs(coeffs_c)**2))
            if norm > 0:
                coeffs_c /= norm

            y = np.zeros(2 * len(modes_list), dtype=np.float32)
            y[0::2] = coeffs_c.real
            y[1::2] = coeffs_c.imag
            Y_list.append(y)
            
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    print(f"Loaded {len(X)} samples — X:{X.shape}  Y:{Y.shape}")
    return X, Y

if USE_H5_DATA:
    X_all, Y_all = load_h5_dataset(H5_FOLDER, MODES)
    split = int(len(X_all) * 0.9)
    X_train, Y_train = X_all[:split], Y_all[:split]
    X_val,   Y_val   = X_all[split:], Y_all[split:]
else:
    X_train, Y_train = generate_dataset(100_000, N_RES)
    X_val,   Y_val   = generate_dataset(10_000,  N_RES)


fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i in range(4):
    c = Y_train[i, 0::2] + 1j*Y_train[i, 1::2]
    axes[0, i].imshow(X_train[i, ..., 0], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, i].set_title(f"Input [{i}]")
    axes[1, i].imshow(synthesize_phase(c, X_train.shape[1]),
                     cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, i].set_title("Re-synth from label")
plt.suptitle("Sanity check — rows must look identical for h5 data to be usable")
plt.tight_layout(); plt.savefig("sanity_check.png", dpi=150); plt.show()


class JAXL2Norm(lay.Layer):
    def __init__(self, epsilon=1e-8, **kw):
        super().__init__(**kw); self.epsilon = epsilon
    def call(self, x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + self.epsilon)
    def get_config(self):
        cfg = super().get_config(); cfg['epsilon'] = self.epsilon; return cfg

# Phase-invariant MSE (aligns global phase before comparing)
def phase_invariant_mse(y_true, y_pred):
    c_true = y_true[:, 0::2] + 1j*y_true[:, 1::2]
    c_pred = y_pred[:, 0::2] + 1j*y_pred[:, 1::2]
    dot    = jnp.sum(jnp.conj(c_pred) * c_true, axis=-1, keepdims=True)
    align  = dot / (jnp.abs(dot) + 1e-8)
    diff   = c_true - c_pred * align
    
    return jnp.mean(jnp.abs(diff)**2)

mode_model = Sequential([
    lay.Input(shape=(N_RES, N_RES, 1)),
    lay.Conv2D(64,  3, activation="elu", padding="same"),
    lay.Conv2D(64,  3, activation="elu", padding="same"),
    lay.AvgPool2D((2, 2)),
    lay.Conv2D(128, 3, activation="elu", padding="same"),
    lay.Conv2D(128, 3, activation="elu", padding="same"),
    lay.AvgPool2D((2, 2)),
    lay.Conv2D(256, 3, activation="elu", padding="same"),
    lay.Conv2D(256, 3, activation="elu", padding="same"),
    lay.GlobalAveragePooling2D(),
    lay.Dense(256, activation="elu"),
    lay.Dropout(0.2),
    lay.Dense(2 * N_COEFF),
    JAXL2Norm(name="coeffs"),
])

mode_model.compile(optimizer=Adam(1e-4, amsgrad=True), loss=phase_invariant_mse)
mode_model.summary()

history = None
try:
    history = mode_model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),   # use explicit val set, not split
        epochs=50, batch_size=64,
    )
except KeyboardInterrupt:
    print("Training interrupted.")



coeffs_pred = mode_model.predict(X_val)
os.makedirs("output_images", exist_ok=True)


mode_labels = [f"p{p},l{l}" for p, l in MODES]
x_pos = np.arange(len(MODES))
width = 0.35

def phase_invariant_fidelity(c_true, c_pred):
    """Overlap |<c_true|c_pred>|^2 after global phase alignment. 1.0 = perfect."""
    dot = np.sum(np.conj(c_pred) * c_true)
    return float(np.abs(dot)**2)

AMP_THRESHOLD = 0.05  # mask phase bars below this amplitude

def align_global_phase(c_true, c_pred):
    """Rotate c_pred to minimize MSE with c_true (removes global phase ambiguity)."""
    dot = np.sum(np.conj(c_pred) * c_true)
    align = dot / (np.abs(dot) + 1e-8)
    return c_pred * align  # aligned prediction

for i in [0, 1, 2, 3]:
    c_true = Y_val[i, 0::2] + 1j * Y_val[i, 1::2]
    c_pred_raw = coeffs_pred[i, 0::2] + 1j * coeffs_pred[i, 1::2]
    c_pred = align_global_phase(c_true, c_pred_raw)
    
    phase_input = X_val[i, ..., 0]
    phase_true  = synthesize_phase(c_true, N_RES)
    phase_pred  = synthesize_phase(c_pred, N_RES)
    fidelity    = phase_invariant_fidelity(c_true, c_pred)

    # Set up the dashboard 
    fig = plt.figure(figsize=(20, 11))
    kw  = dict(cmap='twilight', vmin=-np.pi, vmax=np.pi)

    #row 1: phase images (3 columns) 
    for col, (data, title) in enumerate([
        (phase_input, "Input (Hardware Sim)"),
        (phase_true,  "Re-synth TRUE"),
        (phase_pred,  "Re-synth PREDICTED"),
    ], start=1):
        ax = plt.subplot(2, 3, col)
        im = ax.imshow(data, **kw)
        
       
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('rad', size=14)
        cbar.ax.tick_params(labelsize=12)
        
        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xlabel("px", fontsize=14); ax.set_ylabel("px", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)

    #amplitude bars (2 columns) 
    ax_amp = plt.subplot(2, 2, 3)
    ax_amp.bar(x_pos - width/2, np.abs(c_true), width,
               label='True',      color='navy')
    ax_amp.bar(x_pos + width/2, np.abs(c_pred), width,
               label='Predicted', color='darkorange', alpha=0.85)
    ax_amp.axhline(AMP_THRESHOLD, color='gray', ls='--', lw=1.5,
                   label=f'Mask threshold ({AMP_THRESHOLD})')
    
    
    for j in range(len(MODES)):
        err = np.abs(c_pred)[j] - np.abs(c_true)[j]
        if abs(err) > 0.02:
            ax_amp.text(x_pos[j] + width/2, np.abs(c_pred)[j] + 0.01,
                        f'{err:+.2f}', ha='center', fontsize=10, color='red', fontweight='bold')
            
    ax_amp.set_title("Amplitude per Mode", fontsize=15)
    ax_amp.set_xticks(x_pos)
    ax_amp.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=14)
    ax_amp.set_ylabel("Amplitude", fontsize=16)
    ax_amp.tick_params(axis='y', labelsize=12)
    ax_amp.legend(fontsize=14)

    #phase bars (masked) (2 columns)
    ax_ph = plt.subplot(2, 2, 4)
    mask              = np.abs(c_true) >= AMP_THRESHOLD
    phase_true_masked = np.where(mask, np.angle(c_true), np.nan)
    phase_pred_masked = np.where(mask, np.angle(c_pred), np.nan)
    
    ax_ph.bar(x_pos - width/2, phase_true_masked, width,
              label='True Phase',      color='navy')
    ax_ph.bar(x_pos + width/2, phase_pred_masked, width,
              label='Predicted Phase', color='darkorange', alpha=0.85)
              
    ax_ph.set_title(f"Phase per Mode [rad] (masked where |true amp| < {AMP_THRESHOLD})", fontsize=15)
    ax_ph.set_xticks(x_pos)
    ax_ph.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=14)
    ax_ph.set_ylim(-np.pi, np.pi)
    ax_ph.set_ylabel("Phase [rad]", fontsize=16)
    ax_ph.tick_params(axis='y', labelsize=12)
    ax_ph.legend(fontsize=14)

    
    quality = 'excellent' if fidelity > 0.98 else 'good' if fidelity > 0.90 else 'needs training'
    plt.suptitle(
        f"Validation sample ConfigA {i}  —  Fidelity {fidelity:.3f}  ({quality})",
        fontsize=22, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(f"output_images/ConfigA_diagnostic_{i}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

print("Done!")
