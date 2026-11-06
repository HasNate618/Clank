{
  description = "Clank — a simulated Unitree G1 plays Job Simulator VR with no headset";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  # OpenComposite is a C++ .so that Proton's vrclient bridge dlopen()s inside the
  # Steam runtime. A build from a nixpkgs whose glibc is newer than the runtime's
  # references GLIBC_2.43 symbols the runtime cannot satisfy, so the bridge falls
  # back to wineopenxr and every OpenVR title fails with VRInitError 105 (as Job
  # Simulator did). Build just that package from a nixpkgs whose glibc is at or
  # below the runtime's (this rev is 2.42, matching Steam Linux Runtime 4).
  inputs.nixpkgs-compat.url = "github:NixOS/nixpkgs/0954f7ee2f6bb3dc7d4e3d0d8bcb8fd4bde4cfc5";

  outputs =
    { nixpkgs, nixpkgs-compat, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # Beyley's Monado fork (PSVR2 HMD + 6DOF controllers), built from the compat
      # pin for the same reason as opencomposite: a library built against a newer
      # glibc cannot be loaded inside the Steam Linux Runtime. The patched remote
      # HMD geometry lives in the service this repo runs; keeping it here instead
      # of /etc/nixos puts the VR stack in one place.
      monadoFor =
        system:
        let
          pkgsCompat = nixpkgs-compat.legacyPackages.${system};
        in
        pkgsCompat.monado.overrideAttrs (old: {
          pname = "monado-psvr2";
          version = "psvr2-work-steamvr-2026-06-04";
          src = pkgsCompat.fetchFromGitLab {
            domain = "gitlab.freedesktop.org";
            owner = "Beyley";
            repo = "monado";
            rev = "47f9d39e686d6bbd7dad914b4dbe21f784c41911";
            hash = "sha256-APEA1ivUGTrchePh1XJXycxB9/IRze7ltCPH0qkvAE4=";
          };
          buildInputs = (old.buildInputs or [ ]) ++ (with pkgsCompat; [
            hidapi
            libuvc
            libjpeg
          ]);
          # Upstream nixpkgs patches do not apply to this fork.
          #
          # The remote HMD falls back to Monado's default XR_KHR_visibility_mask,
          # whose hidden mesh clips the lens corners at 25%. Job Simulator stencils
          # those corners to black, so the eye render (and any depth read from it)
          # has a rounded rectangle instead of the full view the flat window shows.
          # Return no hidden area and a full visible quad so the game renders the
          # whole rectangle.
          patches = [ ./spike/e5/monado-no-visibility-mask.patch ];
          postPatch = (old.postPatch or "") + ''
            # The dashboard previews at 30 fps while the game runs at 60 Hz, but
            # Monado's null compositor hardcodes 20 Hz.
            #
            # The compositor, not the remote HMD, is what the app asks for its
            # recommended view size, and it ships a 320x240 recommendation capped
            # at 1920x1080. OXR_VIEWPORT_SCALE_PERCENTAGE multiplies the 320x240,
            # so even 200% was a 640x480 render upscaled into a 1080p window. A
            # 320x240 buffer also has a 4:3 aspect against a 16:9 frustum, which
            # stretches the image. Recommend true 1080p per eye and raise the cap
            # so supersampling is possible.
            substituteInPlace src/xrt/compositor/null/null_compositor.c \
              --replace-fail 'U_TIME_1S_IN_NS / 20' 'U_TIME_1S_IN_NS / 60' \
              --replace-fail 'static const uint64_t RECOMMENDED_VIEW_WIDTH = 320;' 'static const uint64_t RECOMMENDED_VIEW_WIDTH = 1920;' \
              --replace-fail 'static const uint64_t RECOMMENDED_VIEW_HEIGHT = 240;' 'static const uint64_t RECOMMENDED_VIEW_HEIGHT = 1080;' \
              --replace-fail 'static const uint64_t MAX_VIEW_WIDTH = 1920;' 'static const uint64_t MAX_VIEW_WIDTH = 3840;' \
              --replace-fail 'static const uint64_t MAX_VIEW_HEIGHT = 1080;' 'static const uint64_t MAX_VIEW_HEIGHT = 2160;'
            # The remote HMD's display geometry and per-eye FOV are hardcoded here.
            # A 1920x1080 display split side by side gives a portrait 960x1080 eye,
            # so a 1080p view is impossible and the mirror is upscaled; 3840x1080
            # splits into two true 1920x1080 eyes. The vertical FOV is derived from
            # the physical display, and 0.13 x 0.07 is a portrait 0.93 per-eye
            # aspect against a 16:9 buffer, so the frustum does not match the image
            # and the view is distorted. 0.0366 keeps 16:9 per eye; 140 degrees is
            # then ~114.3 degrees vertically, which is deliberately very wide.
            substituteInPlace src/xrt/drivers/remote/r_hmd.c \
              --replace-fail 'info.display.w_pixels = 1920;' 'info.display.w_pixels = 3840;' \
              --replace-fail 'info.display.h_meters = 0.07f;' 'info.display.h_meters = 0.0366f;' \
              --replace-fail 'info.lens_vertical_position_meters = 0.07f / 2.0f;' 'info.lens_vertical_position_meters = 0.0366f / 2.0f;' \
              --replace-fail 'info.fov[0] = 85.0f * (M_PI / 180.0f);' 'info.fov[0] = 140.0f * (M_PI / 180.0f);' \
              --replace-fail 'info.fov[1] = 85.0f * (M_PI / 180.0f);' 'info.fov[1] = 140.0f * (M_PI / 180.0f);'
          '';
          # RenderDoc integration makes the OpenXR client call
          # RENDERDOC_GetAPI()/StartFrameCapture()/EndFrameCapture() around every
          # xrBeginFrame/xrEndFrame when librenderdoc.so is loaded, i.e. whenever
          # our capture layer is active. That writes a full capture per frame to
          # the container's RAM-backed /tmp until the game runs out of memory.
          # We trigger captures explicitly through RenderDoc's target control
          # instead, which does not need this integration.
          cmakeFlags =
            nixpkgs.lib.remove "-DXRT_HAVE_STEAM:BOOL=TRUE" (old.cmakeFlags or [ ])
            ++ [ "-DXRT_FEATURE_RENDERDOC:BOOL=OFF" ];
        });

    in
    {
      packages = forAllSystems (system: {
        monado = monadoFor system;
        default = monadoFor system;
      });

      # The app itself is standard-library-only Python; what this shell pins is
      # the host tooling the launch path needs (nested Weston VNC, websockify
      # and noVNC, plus optional magick and mpv for --preview-smoke).
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          # Only opencomposite comes from the older pin; everything else stays current.
          pkgsCompat = nixpkgs-compat.legacyPackages.${system};
          monado = monadoFor system;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.weston
              pkgs.imagemagick
              pkgs.mpv
              pkgs.novnc
              pkgs.python3Packages.websockify
              # Frame capture for the depth spike: qrenderdoc runs the replay-side
              # Python that reads the game's D32S8 attachment. The Vulkan layer is
              # injected only into the Proton container by spike/e4/depth_capture.py.
              # Take it from the compat pin: the current renderdoc needs glibc 2.43,
              # but Steam Linux Runtime 4 has 2.42 and the in-game layer would fail
              # to load. The compat pin's renderdoc 1.45 needs only glibc 2.38.
              pkgsCompat.renderdoc
              # Build the depth-stream Vulkan layer (spike/e5) and regenerate its
              # SPIR-V if the shader changes. The committed .spv/.h mean glslang is
              # only needed for edits.
              pkgs.vulkan-headers
              pkgs.glslang
              # Sends the fullscreen client message weston's xwm needs; the game's
              # own _NET_WM_STATE_FULLSCREEN is ignored, so it would stay windowed.
              pkgs.xdotool
              # OpenVR -> OpenXR shim. Proton's C:\vrclient bridge loads its
              # bin/linux64/vrclient.so when VR_OVERRIDE points here. xrizer panics
              # on Job Simulator's action manifest; OpenComposite does not.
              pkgsCompat.opencomposite
              # The patched Monado this repo runs; its path reaches the app through
              # CLANK_PATHS_MONADO_BIN below.
              monado
            ];
            shellHook = ''
              # Hand the shim path to the app through its documented env override,
              # so config.toml need not hard-code a /nix/store path.
              export CLANK_PATHS_VR_OVERRIDE="''${CLANK_PATHS_VR_OVERRIDE:-${pkgsCompat.opencomposite}/lib/opencomposite}"
              # Run the repo-built Monado (patched remote HMD, Steam-runtime
              # compatible) instead of the NixOS wrapper. Only the service needs
              # patching: the display geometry and FOV live there, and the host's
              # client library is the same Monado version.
              export CLANK_PATHS_MONADO_BIN="''${CLANK_PATHS_MONADO_BIN:-${monado}/bin/monado-service}"
            '';
          };
        }
      );

      # Only clank/, tests/ and spike/c0/ are copied in: the rest of spike/
      # holds gigabytes of vendored upstreams and weights the tests never touch.
      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default =
            pkgs.runCommand "clank-tests"
              {
                nativeBuildInputs = [ pkgs.python3 ];
              }
              ''
                cp -r ${./clank} clank
                cp -r ${./tests} tests
                # The manual-mode tests load the real pose client from here.
                mkdir -p spike
                cp -r ${./spike/c0} spike/c0
                # test_vnc_auth imports the grabber to exercise its RSA-AES code.
                cp ${./spike/vnc_grab.py} spike/vnc_grab.py
                # run.sh's launcher and its dry-run tests, no layer library needed.
                cp ${./run.sh} run.sh
                mkdir -p spike/e5
                cp ${./spike/e5/launch.py} spike/e5/launch.py
                chmod -R u+w clank tests spike
                # The orchestrator puts its Unix sockets under XDG_RUNTIME_DIR and
                # the sandbox has none, so give it one that exists.
                export XDG_RUNTIME_DIR=$PWD/.xdg-runtime
                mkdir -p "$XDG_RUNTIME_DIR"
                export PYTHONDONTWRITEBYTECODE=1
                python3 -m unittest discover -s tests -t .
                touch $out
              '';
        }
      );
    };
}
