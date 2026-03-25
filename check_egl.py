from OpenGL.EGL import eglGetDisplay, EGL_DEFAULT_DISPLAY, eglInitialize, eglTerminate
from OpenGL import EGL


def check_egl_support():
    try:
        # 尝试获取默认显示设备
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if display == EGL.EGL_NO_DISPLAY:
            print("❌ EGL GetDisplay failed. No display found.")
            return

        # 初始化 EGL
        major, minor = EGL.EGLint(), EGL.EGLint()
        if not eglInitialize(display, major, minor):
            print("❌ EGL Initialize failed.")
            return

        print(f"✅ EGL Initialized successfully. Version: {major.value}.{minor.value}")

        # 检查是否支持 CUDA/NVIDIA 扩展
        extensions = EGL.eglQueryString(display, EGL.EGL_EXTENSIONS)
        if (
            b"EGL_NV_device_cuda" in extensions
            or b"EGL_EXT_platform_device" in extensions
        ):
            print("✅ NVIDIA EGL extensions found (Suitable for headless rendering).")
        else:
            print(
                "⚠️ NVIDIA EGL extensions NOT found. You might fallback to software rendering."
            )

        eglTerminate(display)

    except Exception as e:
        print(f"❌ Error checking EGL: {e}")
        print("Tip: Ensure 'pyopengl' is installed (pip install pyopengl).")


if __name__ == "__main__":
    check_egl_support()
