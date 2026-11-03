// Clank depth-stream Vulkan layer.
//
// Injected only into the game's Proton container. It finds the game's scene
// depth attachment (3840x1080 D32S8, 4x MSAA), samples the left-eye region into
// a 480x270 float32 metric depth buffer with a compute shader, and publishes
// that buffer plus an aligned colour copy into a shared mmap file.
//
// Build: see build.sh. The layer is deliberately self-contained: it uses the
// Vulkan dispatch chain directly and does not link against libvulkan.

#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "depth_stream_spv.h"

#define LAYER_NAME "VK_LAYER_CLANK_depth_stream"
#define MAX_SLOTS 3
#define HEADER_MAGIC 0x434c4b44u /* "CLKD" */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t dtype; /* 0 = float32 */
    uint32_t eye;
    uint32_t color_width;
    uint32_t color_height;
    uint32_t color_dtype; /* 0 = RGBA8 */
    uint32_t slot;
    uint32_t seq; /* odd = being written, even = stable */
    uint32_t reserved0;
    uint64_t frame_id;
    uint64_t inject_ns;
    uint64_t publish_ns;
    float near_m;
    float far_m;
    uint32_t reserved[16];
} Header;

typedef struct {
    uint32_t src_w, src_h, dst_w, dst_h, eye, has_color, color_w, color_h;
    uint32_t out_color_w, out_color_h;
    float near_m, pad;
} PushParams;

typedef struct InstanceData {
    VkInstance instance;
    PFN_vkGetInstanceProcAddr next_gipa;
    PFN_GetPhysicalDeviceProcAddr next_gpdpa;
    struct InstanceData *next;
} InstanceData;

typedef struct PhysInfo {
    VkPhysicalDevice phys;
    VkInstance instance;
    struct PhysInfo *next;
} PhysInfo;

typedef struct ImageInfo {
    VkImage image;
    uint32_t width, height;
    VkFormat format;
    VkSampleCountFlagBits samples;
    VkImageUsageFlags usage;
    int is_depth;
    int is_color;
    struct ImageInfo *next;
} ImageInfo;

typedef struct SwapInfo {
    VkSwapchainKHR sc;
    uint32_t count;
    VkImage *images;
    uint32_t width, height;
    VkFormat format;
    struct SwapInfo *next;
} SwapInfo;

typedef struct ViewInfo {
    VkImageView view;
    VkImage image;
    struct ViewInfo *next;
} ViewInfo;

typedef struct FramebufferInfo {
    VkFramebuffer fb;
    uint32_t count;
    VkImageView *views;
    struct FramebufferInfo *next;
} FramebufferInfo;

typedef struct RenderPassInfo {
    VkRenderPass rp;
    uint32_t n_atts;
    uint32_t depth_att;
    uint32_t color_att;
    VkImageLayout *final_layouts;
    struct RenderPassInfo *next;
} RenderPassInfo;

typedef struct CbInfo {
    VkCommandBuffer cb;
    VkDevice device;
    uint32_t family;
    struct CbInfo *next;
} CbInfo;

typedef struct PoolInfo {
    VkCommandPool pool;
    uint32_t family;
    struct PoolInfo *next;
} PoolInfo;

typedef struct PassState {
    VkCommandBuffer cb;
    VkImage depth;
    VkImage color;
    VkImageLayout depth_layout;
    VkImageLayout color_layout;
    int color_load;
    int depth_load;
    struct PassState *next;
} PassState;

typedef struct DeviceData {
    VkDevice device;
    VkPhysicalDevice physical;
    VkInstance instance;
    PFN_vkGetDeviceProcAddr next_gdpa;
    PFN_vkGetInstanceProcAddr next_gipa;

    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkCreateFramebuffer CreateFramebuffer;
    PFN_vkDestroyFramebuffer DestroyFramebuffer;
    PFN_vkCreateRenderPass CreateRenderPass;
    PFN_vkDestroyRenderPass DestroyRenderPass;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers FreeCommandBuffers;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkGetDeviceQueue2 GetDeviceQueue2;

    PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
    PFN_vkCmdEndRenderPass CmdEndRenderPass;
    PFN_vkCmdBeginRenderPass2 CmdBeginRenderPass2;
    PFN_vkCmdEndRenderPass2 CmdEndRenderPass2;
    PFN_vkCmdBeginRendering CmdBeginRendering;
    PFN_vkCmdEndRendering CmdEndRendering;
    PFN_vkCmdBeginRenderingKHR CmdBeginRenderingKHR;
    PFN_vkCmdEndRenderingKHR CmdEndRenderingKHR;

    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetPhysicalDeviceSurfaceCapabilitiesKHR;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR;
    PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR;
    PFN_vkDestroySwapchainKHR DestroySwapchainKHR;

    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdUpdateBuffer CmdUpdateBuffer;
    PFN_vkCmdExecuteCommands CmdExecuteCommands;

    int failed_forward;
    int setup_attempts;
    int logged_inject;
    int logged_missing_color;
    int pipeline_ready;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkPipeline pipe;
    VkDescriptorPool dpool;
    VkDescriptorSet dsets[MAX_SLOTS];

    VkBuffer dbuf[MAX_SLOTS], cbuf[MAX_SLOTS], sbuf[MAX_SLOTS];
    VkDeviceMemory dmem[MAX_SLOTS], cmem[MAX_SLOTS], smem[MAX_SLOTS];
    void *dmap[MAX_SLOTS], *cmap[MAX_SLOTS], *smap[MAX_SLOTS];

    VkImage target_depth;
    VkImageView depth_view;
    VkFormat depth_format;
    uint32_t depth_w, depth_h;
    VkImage target_color;
    VkImageView color_view;
    VkFormat color_format;
    uint32_t color_w, color_h;

    int shm_fd;
    void *shm;
    size_t shm_size;
    Header *hdr;
    size_t slot_bytes;
    pthread_t thread;
    int thread_started;
    volatile int stop;
    uint32_t last_seq[MAX_SLOTS];
    uint32_t slot_frame[MAX_SLOTS];
    uint64_t slot_inject_ns[MAX_SLOTS];

    uint32_t dst_w, dst_h, eye;
    uint32_t color_scale, out_w, out_h;
    int want_color;
    float near_m, far_m;
    uint32_t seq_counter;
    uint64_t frame_counter;

    struct DeviceData *next;
} DeviceData;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static InstanceData *g_instances;
static PhysInfo *g_phys;
static DeviceData *g_devices;
static VkInstance g_last_instance;
static FILE *g_log;
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

static void logmsg(const char *fmt, ...)
{
    pthread_mutex_lock(&g_log_mu);
    if (!g_log) {
        const char *p = getenv("DEPTH_STREAM_LOG");
        if (!p) p = "/tmp/clank-depth-stream.log";
        g_log = fopen(p, "a");
        if (!g_log) { pthread_mutex_unlock(&g_log_mu); return; }
        setvbuf(g_log, NULL, _IOLBF, 0);
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
    pthread_mutex_unlock(&g_log_mu);
}

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static InstanceData *find_instance(VkInstance inst)
{
    for (InstanceData *it = g_instances; it; it = it->next)
        if (it->instance == inst) return it;
    return NULL;
}

static DeviceData *find_device(VkDevice dev)
{
    for (DeviceData *it = g_devices; it; it = it->next)
        if (it->device == dev) return it;
    return NULL;
}

static CbInfo *g_cbs;

static DeviceData *cb_device(VkCommandBuffer cb)
{
    DeviceData *dd = NULL;
    pthread_mutex_lock(&g_mu);
    for (CbInfo *it = g_cbs; it; it = it->next)
        if (it->cb == cb) { dd = find_device(it->device); break; }
    pthread_mutex_unlock(&g_mu);
    return dd;
}

static ImageInfo *find_image(DeviceData *dd, VkImage img)
{
    (void)dd;
    extern ImageInfo *g_images;
    for (ImageInfo *it = g_images; it; it = it->next)
        if (it->image == img) return it;
    return NULL;
}

ImageInfo *g_images;
SwapInfo *g_swaps;
ViewInfo *g_views;
FramebufferInfo *g_fbs;
RenderPassInfo *g_rps;
PoolInfo *g_pools;
PassState *g_passes;

static VkImage view_to_image(VkImageView view)
{
    for (ViewInfo *it = g_views; it; it = it->next)
        if (it->view == view) return it->image;
    return VK_NULL_HANDLE;
}

static uint32_t pool_family(VkCommandPool pool)
{
    for (PoolInfo *it = g_pools; it; it = it->next)
        if (it->pool == pool) return it->family;
    return 0;
}

static PassState *pass_get(VkCommandBuffer cb)
{
    for (PassState *it = g_passes; it; it = it->next)
        if (it->cb == cb) return it;
    return NULL;
}

static PassState *pass_ensure(VkCommandBuffer cb)
{
    PassState *ps = pass_get(cb);
    if (!ps) {
        ps = calloc(1, sizeof(*ps));
        if (!ps) return NULL;
        ps->cb = cb;
        ps->next = g_passes;
        g_passes = ps;
    }
    return ps;
}


static int create_host_buffer(DeviceData *dd, VkDeviceSize size, VkBufferUsageFlags usage,
                              VkBuffer *buf, VkDeviceMemory *mem, void **map)
{
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    PFN_vkCreateBuffer pCreate = (PFN_vkCreateBuffer)dd->next_gdpa(dd->device, "vkCreateBuffer");
    PFN_vkGetBufferMemoryRequirements pReq = (PFN_vkGetBufferMemoryRequirements)dd->next_gdpa(dd->device, "vkGetBufferMemoryRequirements");
    PFN_vkAllocateMemory pAlloc = (PFN_vkAllocateMemory)dd->next_gdpa(dd->device, "vkAllocateMemory");
    PFN_vkBindBufferMemory pBind = (PFN_vkBindBufferMemory)dd->next_gdpa(dd->device, "vkBindBufferMemory");
    PFN_vkMapMemory pMap = (PFN_vkMapMemory)dd->next_gdpa(dd->device, "vkMapMemory");
    VkResult cr = pCreate(dd->device, &bi, NULL, buf);
    if (cr != VK_SUCCESS) { logmsg("create_host_buffer: CreateBuffer failed %d size=%llu", (int)cr, (unsigned long long)size); return 0; }
    VkMemoryRequirements req;
    memset(&req, 0, sizeof(req));
    pReq(dd->device, *buf, &req);
    logmsg("create_host_buffer: buf=%p create=%d req.size=%llu align=%llu bits=0x%x (pCreate=%p pReq=%p)",
           (void *)*buf, (int)cr, (unsigned long long)req.size, (unsigned long long)req.alignment,
           req.memoryTypeBits, (void *)pCreate, (void *)pReq);
    if (req.size == 0) return 0;
    VkPhysicalDeviceMemoryProperties mp;
    dd->GetPhysicalDeviceMemoryProperties(dd->physical, &mp);
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    int allocated = 0;
    VkResult last = VK_ERROR_UNKNOWN;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        if (!(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        ai.memoryTypeIndex = i;
        last = pAlloc(dd->device, &ai, NULL, mem);
        if (last == VK_SUCCESS) { allocated = 1; break; }
        logmsg("create_host_buffer: AllocateMemory type %u failed %d (size=%llu bits=0x%x flags=0x%x)",
               i, (int)last, (unsigned long long)req.size, req.memoryTypeBits,
               (unsigned)mp.memoryTypes[i].propertyFlags);
    }
    if (!allocated) {
        logmsg("create_host_buffer: no host-visible allocation worked (last=%d size=%llu bits=0x%x)",
               (int)last, (unsigned long long)req.size, req.memoryTypeBits);
        return 0;
    }
    if (pBind(dd->device, *buf, *mem, 0) != VK_SUCCESS) { logmsg("create_host_buffer: BindBufferMemory failed"); return 0; }
    if (pMap(dd->device, *mem, 0, VK_WHOLE_SIZE, 0, map) != VK_SUCCESS) { logmsg("create_host_buffer: MapMemory failed"); return 0; }
    memset(*map, 0, (size_t)size);
    return 1;
}

static VkImageView make_view(DeviceData *dd, VkImage image, VkFormat fmt,
                             VkImageAspectFlags aspect)
{
    VkImageViewCreateInfo vi;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    VkImageView v = VK_NULL_HANDLE;
    if (dd->CreateImageView(dd->device, &vi, NULL, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    return v;
}


static void publish_slot(DeviceData *dd, int slot, uint32_t seq);

static void *publisher_main(void *arg)
{
    DeviceData *dd = arg;
    // The header seq must be one monotonic counter, and only the newest slot
    // may be published. Publishing each changed slot in slot order lets the
    // header step backwards (and go even only briefly), so a reader alternates
    // between older and newer frames: visible as flashing objects. seq is a
    // global counter in inject(), so the maximum is the newest frame.
    uint32_t publish_seq = 0;
    uint32_t last_gpu_seq = 0;
    while (!dd->stop) {
        uint32_t best = 0;
        int best_slot = -1;
        for (int s = 0; s < MAX_SLOTS; s++) {
            uint32_t seq = __atomic_load_n((uint32_t *)dd->smap[s], __ATOMIC_ACQUIRE);
            if (seq > best) {
                best = seq;
                best_slot = s;
            }
        }
        if (best_slot >= 0 && best != last_gpu_seq) {
            last_gpu_seq = best;
            publish_slot(dd, best_slot, ++publish_seq);
        }
        struct timespec ts = {0, 500000}; /* 0.5 ms */
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static int configure(DeviceData *dd)
{
    if (dd->dst_w) return 1;
    dd->dst_w = 480; dd->dst_h = 270; dd->eye = 0;
    const char *v;
    if ((v = getenv("DEPTH_STREAM_WIDTH"))) dd->dst_w = (uint32_t)strtoul(v, NULL, 10);
    if ((v = getenv("DEPTH_STREAM_HEIGHT"))) dd->dst_h = (uint32_t)strtoul(v, NULL, 10);
    if ((v = getenv("DEPTH_STREAM_EYE"))) dd->eye = (uint32_t)strtoul(v, NULL, 10);
    dd->near_m = 0.00992f;
    if ((v = getenv("DEPTH_STREAM_NEAR"))) dd->near_m = strtof(v, NULL);
    dd->far_m = 0.0f;
    dd->want_color = 1;
    if ((v = getenv("DEPTH_STREAM_COLOR"))) dd->want_color = atoi(v) != 0;
    dd->color_scale = 2;
    if ((v = getenv("DEPTH_STREAM_COLOR_SCALE")))
        dd->color_scale = (uint32_t)strtoul(v, NULL, 10);
    if (dd->color_scale < 1) dd->color_scale = 1;
    if (dd->color_scale > 4) dd->color_scale = 4;
    dd->out_w = dd->dst_w * dd->color_scale;
    dd->out_h = dd->dst_h * dd->color_scale;
    return dd->dst_w != 0 && dd->dst_h != 0;
}

static int start_shared(DeviceData *dd)
{
    if (dd->hdr) return 1;
    const char *path = getenv("DEPTH_STREAM_PATH");
    char def[512];
    if (!path || !*path) {
        const char *home = getenv("HOME");
        snprintf(def, sizeof(def), "%s/.local/state/Clank/depth-stream.bin",
                 home ? home : "/tmp");
        path = def;
    }
    size_t pix = (size_t)dd->dst_w * dd->dst_h;
    size_t col_pix = (size_t)dd->out_w * dd->out_h;
    dd->slot_bytes = pix * 4 + (dd->want_color ? col_pix * 4 : 0);
    dd->shm_size = sizeof(Header) + (size_t)MAX_SLOTS * dd->slot_bytes;
    dd->shm_size = (dd->shm_size + 4095) & ~(size_t)4095;

    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) { logmsg("depth-stream: open %s: %s", path, strerror(errno)); return 0; }
    if (ftruncate(fd, (off_t)dd->shm_size) != 0) {
        logmsg("depth-stream: ftruncate: %s", strerror(errno));
        close(fd);
        return 0;
    }
    void *p = mmap(NULL, dd->shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        logmsg("depth-stream: mmap: %s", strerror(errno));
        close(fd);
        return 0;
    }
    dd->shm_fd = fd;
    dd->shm = p;
    dd->hdr = (Header *)p;
    memset(dd->hdr, 0, sizeof(*dd->hdr));
    dd->hdr->magic = HEADER_MAGIC;
    dd->hdr->version = 1;
    dd->hdr->width = dd->dst_w;
    dd->hdr->height = dd->dst_h;
    dd->hdr->dtype = 0;
    dd->hdr->eye = dd->eye;
    dd->hdr->color_width = dd->want_color ? dd->out_w : 0;
    dd->hdr->color_height = dd->want_color ? dd->out_h : 0;
    dd->hdr->color_dtype = 0;
    dd->hdr->near_m = dd->near_m;
    dd->hdr->far_m = dd->far_m;
    __atomic_store_n(&dd->hdr->seq, 0, __ATOMIC_RELEASE);

    for (int s = 0; s < MAX_SLOTS; s++) dd->last_seq[s] = 0;
    if (pthread_create(&dd->thread, NULL, publisher_main, dd) != 0) {
        logmsg("depth-stream: pthread_create failed");
        return 0;
    }
    dd->thread_started = 1;
    logmsg("depth-stream: shared %s (%ux%u eye=%u near=%.6f color=%dx%u scale=%u)",
           path, dd->dst_w, dd->dst_h, dd->eye, dd->near_m,
           dd->want_color ? dd->out_w : 0, dd->want_color ? dd->out_h : 0,
           dd->color_scale);
    return 1;
}

static void publish_slot(DeviceData *dd, int slot, uint32_t publish_seq)
{
    if (!dd->hdr || publish_seq == 0) return;
    uint32_t seq = publish_seq;
    uint8_t *base = (uint8_t *)dd->shm + sizeof(Header) + (size_t)slot * dd->slot_bytes;
    /* Seqlock: odd means a write is in progress. */
    __atomic_store_n(&dd->hdr->slot, (uint32_t)slot, __ATOMIC_RELEASE);
    __atomic_store_n(&dd->hdr->seq, seq * 2u + 1u, __ATOMIC_RELEASE);
    size_t pix = (size_t)dd->dst_w * dd->dst_h;
    size_t col_pix = (size_t)dd->out_w * dd->out_h;
    memcpy(base, dd->dmap[slot], pix * 4);
    if (dd->want_color && dd->cmap[slot])
        memcpy(base + pix * 4, dd->cmap[slot], col_pix * 4);
    dd->hdr->frame_id = dd->slot_frame[slot];
    dd->hdr->inject_ns = dd->slot_inject_ns[slot];
    dd->hdr->publish_ns = now_ns();
    __atomic_store_n(&dd->hdr->seq, seq * 2u + 2u, __ATOMIC_RELEASE);
}

static int ensure_pipeline(DeviceData *dd)
{
    if (dd->pipeline_ready) return 1;
    VkDescriptorSetLayoutBinding b[4];
    memset(b, 0, sizeof(b));
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[3].descriptorCount = 1; b[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dl;
    memset(&dl, 0, sizeof(dl));
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 4;
    dl.pBindings = b;
    if (dd->CreateDescriptorSetLayout(dd->device, &dl, NULL, &dd->dsl) != VK_SUCCESS) { logmsg("pipeline: CreateDescriptorSetLayout failed"); return 0; }

    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushParams);
    VkPipelineLayoutCreateInfo pl;
    memset(&pl, 0, sizeof(pl));
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &dd->dsl;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    if (dd->CreatePipelineLayout(dd->device, &pl, NULL, &dd->pl) != VK_SUCCESS) { logmsg("pipeline: CreatePipelineLayout failed"); return 0; }

    VkShaderModuleCreateInfo sm;
    memset(&sm, 0, sizeof(sm));
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = sizeof(depth_stream_spv);
    sm.pCode = depth_stream_spv;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (dd->CreateShaderModule(dd->device, &sm, NULL, &mod) != VK_SUCCESS) { logmsg("pipeline: CreateShaderModule failed"); return 0; }
    VkComputePipelineCreateInfo cp;
    memset(&cp, 0, sizeof(cp));
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName = "main";
    cp.layout = dd->pl;
    VkResult r = dd->CreateComputePipelines(dd->device, VK_NULL_HANDLE, 1, &cp, NULL, &dd->pipe);
    dd->DestroyShaderModule(dd->device, mod, NULL);
    if (r != VK_SUCCESS) { logmsg("pipeline: CreateComputePipelines failed %d", (int)r); return 0; }

    VkDescriptorPoolSize ps[2];
    ps[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; ps[0].descriptorCount = MAX_SLOTS * 2;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[1].descriptorCount = MAX_SLOTS * 2;
    VkDescriptorPoolCreateInfo dp;
    memset(&dp, 0, sizeof(dp));
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = MAX_SLOTS;
    dp.poolSizeCount = 2;
    dp.pPoolSizes = ps;
    if (dd->CreateDescriptorPool(dd->device, &dp, NULL, &dd->dpool) != VK_SUCCESS) { logmsg("pipeline: CreateDescriptorPool failed"); return 0; }
    VkDescriptorSetAllocateInfo da;
    memset(&da, 0, sizeof(da));
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = dd->dpool;
    da.descriptorSetCount = MAX_SLOTS;
    VkDescriptorSetLayout layouts[MAX_SLOTS];
    for (int i = 0; i < MAX_SLOTS; i++) layouts[i] = dd->dsl;
    da.pSetLayouts = layouts;
    if (dd->AllocateDescriptorSets(dd->device, &da, dd->dsets) != VK_SUCCESS) { logmsg("pipeline: AllocateDescriptorSets failed"); return 0; }

    size_t pix = (size_t)dd->dst_w * dd->dst_h;
    size_t col_pix = (size_t)dd->out_w * dd->out_h;
    for (int s = 0; s < MAX_SLOTS; s++) {
        if (!create_host_buffer(dd, pix * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &dd->dbuf[s], &dd->dmem[s], &dd->dmap[s])) { logmsg("pipeline: depth buffer %d failed", s); return 0; }
        if (dd->want_color && !create_host_buffer(dd, col_pix * 4,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &dd->cbuf[s], &dd->cmem[s], &dd->cmap[s])) { logmsg("pipeline: colour buffer %d failed", s); return 0; }
        if (!create_host_buffer(dd, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &dd->sbuf[s], &dd->smem[s], &dd->smap[s])) { logmsg("pipeline: seq buffer %d failed", s); return 0; }
    }
    dd->pipeline_ready = 1;
    logmsg("depth-stream: compute pipeline ready");
    return 1;
}

static int setup_target(DeviceData *dd, VkImage depth, VkImage color)
{
    if (dd->target_depth == depth && dd->target_color == color && dd->depth_view)
        return 1;
    if (dd->depth_view) {
        dd->DestroyImageView(dd->device, dd->depth_view, NULL);
        dd->depth_view = VK_NULL_HANDLE;
    }
    if (dd->color_view) {
        dd->DestroyImageView(dd->device, dd->color_view, NULL);
        dd->color_view = VK_NULL_HANDLE;
    }
    ImageInfo *di = find_image(dd, depth);
    ImageInfo *ci = find_image(dd, color);
    if (!di || !ci) return 0;
    dd->depth_view = make_view(dd, depth, di->format, VK_IMAGE_ASPECT_DEPTH_BIT);
    dd->color_view = make_view(dd, color, ci->format, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!dd->depth_view || !dd->color_view) return 0;
    dd->target_depth = depth;
    dd->target_color = color;
    dd->depth_format = di->format;
    dd->color_format = ci->format;
    dd->depth_w = di->width;
    dd->depth_h = di->height;
    dd->color_w = ci->width;
    dd->color_h = ci->height;

    for (int s = 0; s < MAX_SLOTS; s++) {
        VkDescriptorImageInfo idd, idc;
        memset(&idd, 0, sizeof(idd)); memset(&idc, 0, sizeof(idc));
        idd.imageView = dd->depth_view; idd.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        idc.imageView = dd->color_view; idc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorBufferInfo bd, bc;
        memset(&bd, 0, sizeof(bd)); memset(&bc, 0, sizeof(bc));
        bd.buffer = dd->dbuf[s]; bd.range = VK_WHOLE_SIZE;
        bc.buffer = dd->cbuf[s]; bc.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w[4];
        memset(w, 0, sizeof(w));
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = dd->dsets[s]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &idd;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = dd->dsets[s]; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &bd;
        w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[2].dstSet = dd->dsets[s]; w[2].dstBinding = 2; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[2].pImageInfo = &idc;
        w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[3].dstSet = dd->dsets[s]; w[3].dstBinding = 3; w[3].descriptorCount = 1;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[3].pBufferInfo = &bc;
        dd->UpdateDescriptorSets(dd->device, 4, w, 0, NULL);
    }
    logmsg("depth-stream: target depth=%p color=%p %ux%u color=%ux%u fmt=%d/%d",
           (void *)depth, (void *)color, dd->depth_w, dd->depth_h,
           dd->color_w, dd->color_h, (int)dd->depth_format, (int)dd->color_format);
    return 1;
}

static void inject(DeviceData *dd, VkCommandBuffer primary, uint32_t family,
                   VkImage depth, VkImage color,
                   VkImageLayout depth_layout, VkImageLayout color_layout)
{
    if (dd->failed_forward) return;
    if (!configure(dd)) { logmsg("depth-stream: bad configuration"); dd->failed_forward = 1; return; }
    if (!ensure_pipeline(dd)) {
        if (dd->setup_attempts++ < 3) logmsg("depth-stream: pipeline setup failed (attempt %d)", dd->setup_attempts);
        return;
    }
    if (!setup_target(dd, depth, color)) return;
    if (!start_shared(dd)) { dd->failed_forward = 1; return; }
    if (depth_layout == VK_IMAGE_LAYOUT_UNDEFINED ||
        color_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        return;
    }

    int slot = (int)(__atomic_fetch_add(&dd->seq_counter, 1, __ATOMIC_RELAXED) % MAX_SLOTS);
    uint32_t seq = __atomic_load_n(&dd->seq_counter, __ATOMIC_RELAXED);
    dd->slot_frame[slot] = (uint32_t)__atomic_fetch_add(&dd->frame_counter, 1, __ATOMIC_RELAXED);
    dd->slot_inject_ns[slot] = now_ns();

    (void)family;

    VkImageMemoryBarrier bars[2];
    memset(bars, 0, sizeof(bars));
    bars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bars[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    bars[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bars[0].oldLayout = depth_layout;
    bars[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bars[0].image = depth;
    bars[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    bars[0].subresourceRange.levelCount = 1;
    bars[0].subresourceRange.layerCount = 1;
    bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bars[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    bars[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bars[1].oldLayout = color_layout;
    bars[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bars[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bars[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bars[1].image = color;
    bars[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bars[1].subresourceRange.levelCount = 1;
    bars[1].subresourceRange.layerCount = 1;
    dd->CmdPipelineBarrier(primary, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           0, NULL, 0, NULL, 2, bars);

    PushParams pc;
    memset(&pc, 0, sizeof(pc));
    pc.src_w = dd->depth_w; pc.src_h = dd->depth_h;
    pc.dst_w = dd->dst_w; pc.dst_h = dd->dst_h;
    pc.eye = dd->eye; pc.has_color = dd->want_color ? 1u : 0u;
    pc.color_w = dd->color_w; pc.color_h = dd->color_h;
    pc.out_color_w = dd->out_w; pc.out_color_h = dd->out_h;
    pc.near_m = dd->near_m;
    dd->CmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, dd->pipe);
    dd->CmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_COMPUTE, dd->pl, 0, 1,
                              &dd->dsets[slot], 0, NULL);
    dd->CmdPushConstants(primary, dd->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t max_w = dd->dst_w > dd->out_w ? dd->dst_w : dd->out_w;
    uint32_t max_h = dd->dst_h > dd->out_h ? dd->dst_h : dd->out_h;
    uint32_t gx = (max_w + 7) / 8, gy = (max_h + 7) / 8;
    dd->CmdDispatch(primary, gx, gy, 1);

    /* Restore the game's layouts, making the compute writes available to the
       transfer stage (for the sequence update) and to the host. */
    bars[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bars[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    bars[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bars[0].newLayout = depth_layout;
    bars[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bars[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    bars[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bars[1].newLayout = color_layout;
    dd->CmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                           0, NULL, 0, NULL, 2, bars);

    VkBufferMemoryBarrier bb;
    memset(&bb, 0, sizeof(bb));
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = dd->dbuf[slot];
    bb.offset = 0;
    bb.size = VK_WHOLE_SIZE;
    dd->CmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0,
                           0, NULL, 1, &bb, 0, NULL);
    if (dd->want_color) {
        VkBufferMemoryBarrier cb2 = bb;
        cb2.buffer = dd->cbuf[slot];
        dd->CmdPipelineBarrier(primary, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT, 0,
                               0, NULL, 1, &cb2, 0, NULL);
    }
    dd->CmdUpdateBuffer(primary, dd->sbuf[slot], 0, sizeof(seq), &seq);
    VkBufferMemoryBarrier sb;
    memset(&sb, 0, sizeof(sb));
    sb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    sb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    sb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    sb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sb.buffer = dd->sbuf[slot];
    sb.offset = 0;
    sb.size = VK_WHOLE_SIZE;
    dd->CmdPipelineBarrier(primary, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0,
                           0, NULL, 1, &sb, 0, NULL);

}

static void begin_pass_common(DeviceData *dd, VkCommandBuffer cb, VkImage depth, VkImage color,
                              VkImageLayout depth_layout, VkImageLayout color_layout,
                              VkAttachmentLoadOp color_load, VkAttachmentLoadOp depth_load)
{
    (void)dd;
    PassState *ps = pass_ensure(cb);
    if (!ps) return;
    ps->depth = depth;
    ps->color = color;
    ps->depth_layout = depth_layout;
    ps->color_layout = color_layout;
    ps->color_load = (int)color_load;
    ps->depth_load = (int)depth_load;
}

static void end_pass_common(VkCommandBuffer cb)
{
    PassState *ps = pass_get(cb);
    if (!ps) return;
    VkImage depth = ps->depth, color = ps->color;
    VkImageLayout dl = ps->depth_layout, cl = ps->color_layout;
    if (depth == VK_NULL_HANDLE || color == VK_NULL_HANDLE) return;
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    ImageInfo *di = find_image(dd, depth);
    if (!di || !di->is_depth) return;
    ImageInfo *ci = find_image(dd, color);
    if (!ci || !ci->is_color) {
        if (!dd->logged_missing_color) {
            dd->logged_missing_color = 1;
            logmsg("pass end: target depth but color %p is not a candidate", (void *)color);
        }
        return;
    }
    // The eye target is written twice: a CLEAR pass for one eye and a LOAD
    // pass that adds the other eye / glow. The LOAD pass holds the final
    // colour, so injecting both makes the preview alternate between dim and
    // bright. Stream the LOAD pass only.
    if (ps->color_load != VK_ATTACHMENT_LOAD_OP_LOAD) return;
    CbInfo *cbi = NULL;
    for (CbInfo *it = g_cbs; it; it = it->next)
        if (it->cb == cb) { cbi = it; break; }
    uint32_t family = cbi ? cbi->family : 0;
    if (!dd->logged_inject) {
        dd->logged_inject = 1;
        logmsg("injecting depth=%p color=%p dl=%d cl=%d", (void *)depth, (void *)color,
               (int)dl, (int)cl);
    }
    inject(dd, cb, family, depth, color, dl, cl);
}

/* ------------------------- Vulkan entry points ------------------------- */

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateInstance(
    const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *alloc, VkInstance *out)
{
    VkLayerInstanceCreateInfo *chain = (VkLayerInstanceCreateInfo *)ci->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_GetPhysicalDeviceProcAddr next_gpdpa = chain->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)next_gipa(NULL, "vkCreateInstance");
    VkResult r = create(ci, alloc, out);
    if (r == VK_SUCCESS) {
        InstanceData *id = calloc(1, sizeof(*id));
        if (id) {
            id->instance = *out;
            id->next_gipa = next_gipa;
            id->next_gpdpa = next_gpdpa;
            pthread_mutex_lock(&g_mu);
            id->next = g_instances;
            g_instances = id;
            g_last_instance = *out;
            pthread_mutex_unlock(&g_mu);
        }
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_EnumeratePhysicalDevices(
    VkInstance inst, uint32_t *count, VkPhysicalDevice *devs)
{
    InstanceData *id = find_instance(inst);
    if (!id) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkEnumeratePhysicalDevices fn =
        (PFN_vkEnumeratePhysicalDevices)id->next_gipa(inst, "vkEnumeratePhysicalDevices");
    VkResult r = fn(inst, count, devs);
    if (r == VK_SUCCESS && devs) {
        pthread_mutex_lock(&g_mu);
        for (uint32_t i = 0; i < *count; i++) {
            PhysInfo *pi = calloc(1, sizeof(*pi));
            if (pi) { pi->phys = devs[i]; pi->instance = inst; pi->next = g_phys; g_phys = pi; }
        }
        pthread_mutex_unlock(&g_mu);
    }
    return r;
}

static VkInstance instance_for_phys(VkPhysicalDevice phys)
{
    for (PhysInfo *it = g_phys; it; it = it->next)
        if (it->phys == phys) return it->instance;
    return g_last_instance;
}

static void resolve_device_funcs(DeviceData *dd)
{
    PFN_vkGetDeviceProcAddr g = dd->next_gdpa;
#define R(name) dd->name = (PFN_vk##name)g(dd->device, "vk" #name)
    R(DestroyDevice);
    R(CreateImage); R(DestroyImage);
    R(CreateImageView); R(DestroyImageView);
    R(CreateFramebuffer); R(DestroyFramebuffer);
    R(CreateRenderPass); R(DestroyRenderPass);
    R(CreateCommandPool); R(DestroyCommandPool);
    R(AllocateCommandBuffers); R(FreeCommandBuffers);
    R(CmdBeginRenderPass); R(CmdEndRenderPass);
    R(CmdBeginRenderPass2); R(CmdEndRenderPass2);
    R(CmdBeginRendering); R(CmdEndRendering);
    R(CmdBeginRenderingKHR); R(CmdEndRenderingKHR);
    R(CreateShaderModule); R(DestroyShaderModule);
    R(CreateDescriptorSetLayout); R(CreatePipelineLayout);
    R(CreateComputePipelines); R(DestroyPipeline);
    R(CreateDescriptorPool); R(AllocateDescriptorSets); R(UpdateDescriptorSets);
    R(CreateBuffer); R(DestroyBuffer); R(GetBufferMemoryRequirements);
    R(AllocateMemory); R(FreeMemory); R(BindBufferMemory);
    R(MapMemory); R(UnmapMemory);
    R(CmdPipelineBarrier); R(CmdBindPipeline); R(CmdBindDescriptorSets);
    R(CmdPushConstants); R(CmdDispatch); R(CmdUpdateBuffer); R(CmdExecuteCommands);
    R(CreateSwapchainKHR); R(GetSwapchainImagesKHR); R(DestroySwapchainKHR);
#undef R
    dd->GetPhysicalDeviceMemoryProperties =
        (PFN_vkGetPhysicalDeviceMemoryProperties)dd->next_gipa(
            dd->instance, "vkGetPhysicalDeviceMemoryProperties");
    dd->GetPhysicalDeviceSurfaceCapabilitiesKHR =
        (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)dd->next_gipa(
            dd->instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    dd->GetPhysicalDeviceProperties =
        (PFN_vkGetPhysicalDeviceProperties)dd->next_gipa(
            dd->instance, "vkGetPhysicalDeviceProperties");
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateDevice(
    VkPhysicalDevice phys, const VkDeviceCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkDevice *out)
{
    VkLayerDeviceCreateInfo *chain = (VkLayerDeviceCreateInfo *)ci->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    VkInstance inst = instance_for_phys(phys);
    PFN_vkCreateDevice create = (PFN_vkCreateDevice)next_gipa(inst, "vkCreateDevice");
    VkResult r = create(phys, ci, alloc, out);
    if (r != VK_SUCCESS) return r;
    DeviceData *dd = calloc(1, sizeof(*dd));
    if (!dd) return VK_ERROR_OUT_OF_HOST_MEMORY;
    dd->device = *out;
    dd->physical = phys;
    dd->instance = inst;
    dd->next_gipa = next_gipa;
    dd->next_gdpa = next_gdpa;
    pthread_mutex_lock(&g_mu);
    dd->next = g_devices;
    g_devices = dd;
    pthread_mutex_unlock(&g_mu);
    resolve_device_funcs(dd);
    if (dd->GetPhysicalDeviceProperties) {
        VkPhysicalDeviceProperties props;
        dd->GetPhysicalDeviceProperties(dd->physical, &props);
        logmsg("depth-stream: device %p phys=%p '%s' api=%u.%u.%u inst=%p", (void *)*out,
               (void *)phys, props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
               VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion),
               (void *)inst);
    } else {
        logmsg("depth-stream: device %p created (no properties fn)", (void *)*out);
    }
    return VK_SUCCESS;
}

#define RESOLVE(dd, name) ((dd)->name = (PFN_vk##name)(dd)->next_gdpa((dd)->device, "vk" #name))

static PFN_vkCreateImage get_CreateImage(DeviceData *dd)
{
    if (!dd->CreateImage) RESOLVE(dd, CreateImage);
    return dd->CreateImage;
}
static PFN_vkDestroyImage get_DestroyImage(DeviceData *dd)
{
    if (!dd->DestroyImage) RESOLVE(dd, DestroyImage);
    return dd->DestroyImage;
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateImage(
    VkDevice dev, const VkImageCreateInfo *ci, const VkAllocationCallbacks *alloc, VkImage *img)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    VkImageCreateInfo info = *ci;
    int is_depth = info.imageType == VK_IMAGE_TYPE_2D &&
                   info.extent.width == 3840 && info.extent.height == 1080 &&
                   info.format == VK_FORMAT_D32_SFLOAT_S8_UINT &&
                   info.samples == VK_SAMPLE_COUNT_4_BIT &&
                   info.mipLevels == 1 && info.arrayLayers == 1;
    int is_color = info.imageType == VK_IMAGE_TYPE_2D &&
                   info.extent.width == 3840 && info.extent.height == 1080 &&
                   info.samples == VK_SAMPLE_COUNT_4_BIT &&
                   (info.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
                   info.mipLevels == 1 && info.arrayLayers == 1 &&
                   !is_depth;
    if (is_depth || is_color) {
        info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    PFN_vkCreateImage fn = get_CreateImage(dd);
    VkResult r = fn(dev, &info, alloc, img);
    if (r == VK_SUCCESS && info.imageType == VK_IMAGE_TYPE_2D && info.extent.height == 1080)
        logmsg("image %p %ux%u fmt=%d samples=%d usage=0x%x depth=%d color=%d",
               (void *)*img, info.extent.width, info.extent.height, (int)info.format,
               (int)info.samples, (unsigned)info.usage, is_depth, is_color);
    if (r == VK_SUCCESS && (is_depth || is_color)) {
        ImageInfo *ii = calloc(1, sizeof(*ii));
        if (ii) {
            ii->image = *img; ii->width = info.extent.width; ii->height = info.extent.height;
            ii->format = info.format; ii->samples = info.samples; ii->usage = info.usage;
            ii->is_depth = is_depth; ii->is_color = is_color;
            pthread_mutex_lock(&g_mu);
            ii->next = g_images; g_images = ii;
            pthread_mutex_unlock(&g_mu);
            logmsg("depth-stream: candidate image %p %ux%u fmt=%d samples=%d depth=%d color=%d",
                   (void *)*img, ii->width, ii->height, (int)ii->format,
                   (int)ii->samples, is_depth, is_color);
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_DestroyImage(
    VkDevice dev, VkImage img, const VkAllocationCallbacks *alloc)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return;
    pthread_mutex_lock(&g_mu);
    ImageInfo **pp = &g_images;
    while (*pp) {
        if ((*pp)->image == img) { ImageInfo *dead = *pp; *pp = dead->next; free(dead); break; }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_mu);
    PFN_vkDestroyImage fn = get_DestroyImage(dd);
    fn(dev, img, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateImageView(
    VkDevice dev, const VkImageViewCreateInfo *ci, const VkAllocationCallbacks *alloc, VkImageView *view)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    if (!dd->CreateImageView) RESOLVE(dd, CreateImageView);
    VkResult r = dd->CreateImageView(dev, ci, alloc, view);
    if (r == VK_SUCCESS) {
        ViewInfo *vi = calloc(1, sizeof(*vi));
        if (vi) {
            vi->view = *view; vi->image = ci->image;
            pthread_mutex_lock(&g_mu);
            vi->next = g_views; g_views = vi;
            pthread_mutex_unlock(&g_mu);
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_DestroyImageView(
    VkDevice dev, VkImageView view, const VkAllocationCallbacks *alloc)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return;
    pthread_mutex_lock(&g_mu);
    ViewInfo **pp = &g_views;
    while (*pp) {
        if ((*pp)->view == view) { ViewInfo *dead = *pp; *pp = dead->next; free(dead); break; }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_mu);
    if (!dd->DestroyImageView) RESOLVE(dd, DestroyImageView);
    dd->DestroyImageView(dev, view, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateFramebuffer(
    VkDevice dev, const VkFramebufferCreateInfo *ci, const VkAllocationCallbacks *alloc,
    VkFramebuffer *fb)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    if (!dd->CreateFramebuffer) RESOLVE(dd, CreateFramebuffer);
    VkResult r = dd->CreateFramebuffer(dev, ci, alloc, fb);
    if (r == VK_SUCCESS) {
        FramebufferInfo *fi = calloc(1, sizeof(*fi));
        if (fi) {
            fi->fb = *fb; fi->count = ci->attachmentCount;
            fi->views = calloc(ci->attachmentCount, sizeof(VkImageView));
            if (fi->views) memcpy(fi->views, ci->pAttachments, ci->attachmentCount * sizeof(VkImageView));
            pthread_mutex_lock(&g_mu);
            fi->next = g_fbs; g_fbs = fi;
            pthread_mutex_unlock(&g_mu);
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_DestroyFramebuffer(
    VkDevice dev, VkFramebuffer fb, const VkAllocationCallbacks *alloc)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return;
    pthread_mutex_lock(&g_mu);
    FramebufferInfo **pp = &g_fbs;
    while (*pp) {
        if ((*pp)->fb == fb) { FramebufferInfo *dead = *pp; *pp = dead->next; free(dead->views); free(dead); break; }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_mu);
    if (!dd->DestroyFramebuffer) RESOLVE(dd, DestroyFramebuffer);
    dd->DestroyFramebuffer(dev, fb, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateRenderPass(
    VkDevice dev, const VkRenderPassCreateInfo *ci, const VkAllocationCallbacks *alloc,
    VkRenderPass *rp)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    if (!dd->CreateRenderPass) RESOLVE(dd, CreateRenderPass);
    VkResult r = dd->CreateRenderPass(dev, ci, alloc, rp);
    if (r == VK_SUCCESS) {
        RenderPassInfo *ri = calloc(1, sizeof(*ri));
        if (ri) {
            ri->rp = *rp; ri->n_atts = ci->attachmentCount;
            ri->depth_att = UINT32_MAX; ri->color_att = UINT32_MAX;
            ri->final_layouts = calloc(ci->attachmentCount, sizeof(VkImageLayout));
            if (ri->final_layouts) {
                for (uint32_t i = 0; i < ci->attachmentCount; i++)
                    ri->final_layouts[i] = ci->pAttachments[i].finalLayout;
            }
            if (ci->subpassCount > 0) {
                const VkSubpassDescription *sp = &ci->pSubpasses[0];
                if (sp->pDepthStencilAttachment)
                    ri->depth_att = sp->pDepthStencilAttachment->attachment;
                if (sp->colorAttachmentCount > 0)
                    ri->color_att = sp->pColorAttachments[0].attachment;
            }
            pthread_mutex_lock(&g_mu);
            ri->next = g_rps; g_rps = ri;
            pthread_mutex_unlock(&g_mu);
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_DestroyRenderPass(
    VkDevice dev, VkRenderPass rp, const VkAllocationCallbacks *alloc)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return;
    pthread_mutex_lock(&g_mu);
    RenderPassInfo **pp = &g_rps;
    while (*pp) {
        if ((*pp)->rp == rp) { RenderPassInfo *dead = *pp; *pp = dead->next; free(dead->final_layouts); free(dead); break; }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_mu);
    if (!dd->DestroyRenderPass) RESOLVE(dd, DestroyRenderPass);
    dd->DestroyRenderPass(dev, rp, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateCommandPool(
    VkDevice dev, const VkCommandPoolCreateInfo *ci, const VkAllocationCallbacks *alloc,
    VkCommandPool *pool)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    if (!dd->CreateCommandPool) RESOLVE(dd, CreateCommandPool);
    VkResult r = dd->CreateCommandPool(dev, ci, alloc, pool);
    if (r == VK_SUCCESS) {
        PoolInfo *pi = calloc(1, sizeof(*pi));
        if (pi) {
            pi->pool = *pool; pi->family = ci->queueFamilyIndex;
            pthread_mutex_lock(&g_mu);
            pi->next = g_pools; g_pools = pi;
            pthread_mutex_unlock(&g_mu);
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_DestroyCommandPool(
    VkDevice dev, VkCommandPool pool, const VkAllocationCallbacks *alloc)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return;
    pthread_mutex_lock(&g_mu);
    PoolInfo **pp = &g_pools;
    while (*pp) {
        if ((*pp)->pool == pool) { PoolInfo *dead = *pp; *pp = dead->next; free(dead); break; }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_mu);
    if (!dd->DestroyCommandPool) RESOLVE(dd, DestroyCommandPool);
    dd->DestroyCommandPool(dev, pool, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_AllocateCommandBuffers(
    VkDevice dev, const VkCommandBufferAllocateInfo *ai, VkCommandBuffer *cbs)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    if (!dd->AllocateCommandBuffers) RESOLVE(dd, AllocateCommandBuffers);
    VkResult r = dd->AllocateCommandBuffers(dev, ai, cbs);
    if (r == VK_SUCCESS) {
        uint32_t fam = pool_family(ai->commandPool);
        pthread_mutex_lock(&g_mu);
        for (uint32_t i = 0; i < ai->commandBufferCount; i++) {
            CbInfo *ci = calloc(1, sizeof(*ci));
            if (ci) { ci->cb = cbs[i]; ci->device = dev; ci->family = fam; ci->next = g_cbs; g_cbs = ci; }
        }
        pthread_mutex_unlock(&g_mu);
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_FreeCommandBuffers(
    VkDevice dev, VkCommandPool pool, uint32_t count, const VkCommandBuffer *cbs)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return;
    pthread_mutex_lock(&g_mu);
    for (uint32_t i = 0; i < count; i++) {
        CbInfo **pp = &g_cbs;
        while (*pp) {
            if ((*pp)->cb == cbs[i]) { CbInfo *dead = *pp; *pp = dead->next; free(dead); break; }
            pp = &(*pp)->next;
        }
    }
    pthread_mutex_unlock(&g_mu);
    if (!dd->FreeCommandBuffers) RESOLVE(dd, FreeCommandBuffers);
    dd->FreeCommandBuffers(dev, pool, count, cbs);
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdBeginRendering(
    VkCommandBuffer cb, const VkRenderingInfo *ri)
{
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    if (!dd->CmdBeginRendering) RESOLVE(dd, CmdBeginRendering);
    dd->CmdBeginRendering(cb, ri);
    VkImage depth = VK_NULL_HANDLE, color = VK_NULL_HANDLE;
    VkImageLayout dl = VK_IMAGE_LAYOUT_UNDEFINED, cl = VK_IMAGE_LAYOUT_UNDEFINED;
    if (ri->pDepthAttachment && ri->pDepthAttachment->imageView)
        depth = view_to_image(ri->pDepthAttachment->imageView);
    if (ri->pDepthAttachment) dl = ri->pDepthAttachment->imageLayout;
    if (ri->colorAttachmentCount > 0 && ri->pColorAttachments[0].imageView)
        color = view_to_image(ri->pColorAttachments[0].imageView);
    if (ri->colorAttachmentCount > 0) cl = ri->pColorAttachments[0].imageLayout;
    VkAttachmentLoadOp cload = ri->colorAttachmentCount > 0 ? ri->pColorAttachments[0].loadOp : 0;
    VkAttachmentLoadOp dload = ri->pDepthAttachment ? ri->pDepthAttachment->loadOp : 0;
    begin_pass_common(dd, cb, depth, color, dl, cl, cload, dload);
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdEndRendering(VkCommandBuffer cb)
{
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    if (!dd->CmdEndRendering) RESOLVE(dd, CmdEndRendering);
    dd->CmdEndRendering(cb);
    end_pass_common(cb);
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdBeginRenderingKHR(
    VkCommandBuffer cb, const VkRenderingInfo *ri)
{
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    if (!dd->CmdBeginRenderingKHR) RESOLVE(dd, CmdBeginRenderingKHR);
    if (dd->CmdBeginRenderingKHR) dd->CmdBeginRenderingKHR(cb, ri);
    else if (dd->CmdBeginRendering) dd->CmdBeginRendering(cb, ri);
    VkImage depth = VK_NULL_HANDLE, color = VK_NULL_HANDLE;
    VkImageLayout dl = VK_IMAGE_LAYOUT_UNDEFINED, cl = VK_IMAGE_LAYOUT_UNDEFINED;
    if (ri->pDepthAttachment && ri->pDepthAttachment->imageView)
        depth = view_to_image(ri->pDepthAttachment->imageView);
    if (ri->pDepthAttachment) dl = ri->pDepthAttachment->imageLayout;
    if (ri->colorAttachmentCount > 0 && ri->pColorAttachments[0].imageView)
        color = view_to_image(ri->pColorAttachments[0].imageView);
    if (ri->colorAttachmentCount > 0) cl = ri->pColorAttachments[0].imageLayout;
    VkAttachmentLoadOp cload = ri->colorAttachmentCount > 0 ? ri->pColorAttachments[0].loadOp : 0;
    VkAttachmentLoadOp dload = ri->pDepthAttachment ? ri->pDepthAttachment->loadOp : 0;
    begin_pass_common(dd, cb, depth, color, dl, cl, cload, dload);
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdEndRenderingKHR(VkCommandBuffer cb)
{
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    if (!dd->CmdEndRenderingKHR) RESOLVE(dd, CmdEndRenderingKHR);
    if (dd->CmdEndRenderingKHR) dd->CmdEndRenderingKHR(cb);
    else if (dd->CmdEndRendering) dd->CmdEndRendering(cb);
    end_pass_common(cb);
}

static void classic_begin(DeviceData *dd, VkCommandBuffer cb, const VkRenderPassBeginInfo *bi,
                          VkSubpassContents contents)
{
    if (!dd->CmdBeginRenderPass) RESOLVE(dd, CmdBeginRenderPass);
    dd->CmdBeginRenderPass(cb, bi, contents);
    VkImage depth = VK_NULL_HANDLE, color = VK_NULL_HANDLE;
    VkImageLayout dl = VK_IMAGE_LAYOUT_UNDEFINED, cl = VK_IMAGE_LAYOUT_UNDEFINED;
    FramebufferInfo *fi = NULL;
    RenderPassInfo *ri = NULL;
    for (FramebufferInfo *it = g_fbs; it; it = it->next)
        if (it->fb == bi->framebuffer) { fi = it; break; }
    for (RenderPassInfo *it = g_rps; it; it = it->next)
        if (it->rp == bi->renderPass) { ri = it; break; }
    if (fi && ri) {
        if (ri->depth_att != UINT32_MAX && ri->depth_att < fi->count)
            depth = view_to_image(fi->views[ri->depth_att]);
        if (ri->color_att != UINT32_MAX && ri->color_att < fi->count)
            color = view_to_image(fi->views[ri->color_att]);
        if (ri->final_layouts) {
            if (ri->depth_att != UINT32_MAX) dl = ri->final_layouts[ri->depth_att];
            if (ri->color_att != UINT32_MAX) cl = ri->final_layouts[ri->color_att];
        }
    }
    begin_pass_common(dd, cb, depth, color, dl, cl, 0, 0);
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdBeginRenderPass(
    VkCommandBuffer cb, const VkRenderPassBeginInfo *bi, VkSubpassContents contents)
{
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    classic_begin(dd, cb, bi, contents);
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdEndRenderPass(VkCommandBuffer cb)
{
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    if (!dd->CmdEndRenderPass) RESOLVE(dd, CmdEndRenderPass);
    dd->CmdEndRenderPass(cb);
    end_pass_common(cb);
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdBeginRenderPass2(
    VkCommandBuffer cb, const VkRenderPassBeginInfo *bi, const VkSubpassBeginInfo *si)
{
    (void)si;
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    if (!dd->CmdBeginRenderPass2) RESOLVE(dd, CmdBeginRenderPass2);
    if (dd->CmdBeginRenderPass2) {
        dd->CmdBeginRenderPass2(cb, bi, si);
        VkImage depth = VK_NULL_HANDLE, color = VK_NULL_HANDLE;
        VkImageLayout dl = VK_IMAGE_LAYOUT_UNDEFINED, cl = VK_IMAGE_LAYOUT_UNDEFINED;
        FramebufferInfo *fi = NULL; RenderPassInfo *ri = NULL;
        for (FramebufferInfo *it = g_fbs; it; it = it->next)
            if (it->fb == bi->framebuffer) { fi = it; break; }
        for (RenderPassInfo *it = g_rps; it; it = it->next)
            if (it->rp == bi->renderPass) { ri = it; break; }
        if (fi && ri) {
            if (ri->depth_att != UINT32_MAX && ri->depth_att < fi->count)
                depth = view_to_image(fi->views[ri->depth_att]);
            if (ri->color_att != UINT32_MAX && ri->color_att < fi->count)
                color = view_to_image(fi->views[ri->color_att]);
            if (ri->final_layouts) {
                if (ri->depth_att != UINT32_MAX) dl = ri->final_layouts[ri->depth_att];
                if (ri->color_att != UINT32_MAX) cl = ri->final_layouts[ri->color_att];
            }
        }
        begin_pass_common(dd, cb, depth, color, dl, cl, 0, 0);
    }
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_CmdEndRenderPass2(
    VkCommandBuffer cb, const VkSubpassEndInfo *ei)
{
    (void)ei;
    DeviceData *dd = cb_device(cb);
    if (!dd) return;
    if (!dd->CmdEndRenderPass2) RESOLVE(dd, CmdEndRenderPass2);
    if (dd->CmdEndRenderPass2) dd->CmdEndRenderPass2(cb, ei);
    end_pass_common(cb);
}

static SwapInfo *find_swap(VkSwapchainKHR sc)
{
    for (SwapInfo *it = g_swaps; it; it = it->next)
        if (it->sc == sc) return it;
    return NULL;
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_CreateSwapchainKHR(
    VkDevice dev, const VkSwapchainCreateInfoKHR *ci,
    const VkAllocationCallbacks *alloc, VkSwapchainKHR *sc)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    if (!dd->CreateSwapchainKHR) RESOLVE(dd, CreateSwapchainKHR);
    VkSwapchainCreateInfoKHR info = *ci;
    if (dd->GetPhysicalDeviceSurfaceCapabilitiesKHR) {
        VkSurfaceCapabilitiesKHR caps;
        if (dd->GetPhysicalDeviceSurfaceCapabilitiesKHR(dd->physical, info.surface, &caps) == VK_SUCCESS) {
            if (caps.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT)
                info.imageUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            else
                logmsg("swapchain: surface does not support SAMPLED");
        }
    }
    VkResult r = dd->CreateSwapchainKHR(dev, &info, alloc, sc);
    if (r == VK_SUCCESS) {
        SwapInfo *si = calloc(1, sizeof(*si));
        if (si) {
            si->sc = *sc;
            si->width = info.imageExtent.width;
            si->height = info.imageExtent.height;
            si->format = info.imageFormat;
            pthread_mutex_lock(&g_mu);
            si->next = g_swaps; g_swaps = si;
            pthread_mutex_unlock(&g_mu);
            logmsg("swapchain %p %ux%u fmt=%d usage=0x%x", (void *)*sc, si->width, si->height,
                   (int)si->format, (unsigned)info.imageUsage);
        }
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL clank_depth_GetSwapchainImagesKHR(
    VkDevice dev, VkSwapchainKHR sc, uint32_t *count, VkImage *imgs)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return VK_ERROR_INITIALIZATION_FAILED;
    if (!dd->GetSwapchainImagesKHR) RESOLVE(dd, GetSwapchainImagesKHR);
    VkResult r = dd->GetSwapchainImagesKHR(dev, sc, count, imgs);
    if ((r == VK_SUCCESS || r == VK_INCOMPLETE) && imgs && count) {
        pthread_mutex_lock(&g_mu);
        SwapInfo *si = find_swap(sc);
        if (si) {
            free(si->images);
            si->images = calloc(*count, sizeof(VkImage));
            si->count = *count;
            for (uint32_t i = 0; i < *count; i++) {
                si->images[i] = imgs[i];
                logmsg("swapchain image %p %ux%u fmt=%d", (void *)imgs[i], si->width, si->height, (int)si->format);
            }
            logmsg("swapchain images %p count=%u", (void *)sc, *count);
        }
        pthread_mutex_unlock(&g_mu);
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_DestroySwapchainKHR(
    VkDevice dev, VkSwapchainKHR sc, const VkAllocationCallbacks *alloc)
{
    DeviceData *dd = find_device(dev);
    if (dd) {
        pthread_mutex_lock(&g_mu);
        SwapInfo **pp = &g_swaps;
        while (*pp) {
            if ((*pp)->sc == sc) {
                SwapInfo *dead = *pp; *pp = dead->next;
                for (uint32_t i = 0; i < dead->count; i++) {
                    ImageInfo **ip = &g_images;
                    while (*ip) {
                        if ((*ip)->image == dead->images[i]) {
                            ImageInfo *di = *ip; *ip = di->next; free(di); break;
                        }
                        ip = &(*ip)->next;
                    }
                }
                free(dead->images); free(dead);
                break;
            }
            pp = &(*pp)->next;
        }
        pthread_mutex_unlock(&g_mu);
        if (!dd->DestroySwapchainKHR) RESOLVE(dd, DestroySwapchainKHR);
        dd->DestroySwapchainKHR(dev, sc, alloc);
    }
}

static VKAPI_ATTR void VKAPI_CALL clank_depth_DestroyDevice(
    VkDevice dev, const VkAllocationCallbacks *alloc)
{
    DeviceData *dd = find_device(dev);
    if (dd) {
        dd->stop = 1;
        if (dd->thread_started) pthread_join(dd->thread, NULL);
        pthread_mutex_lock(&g_mu);
        DeviceData **pp = &g_devices;
        while (*pp) { if (*pp == dd) { *pp = dd->next; break; } pp = &(*pp)->next; }
        pthread_mutex_unlock(&g_mu);
        if (!dd->DestroyDevice) RESOLVE(dd, DestroyDevice);
        if (dd->DestroyDevice) dd->DestroyDevice(dev, alloc);
        if (dd->shm && dd->shm != MAP_FAILED) munmap(dd->shm, dd->shm_size);
        if (dd->shm_fd >= 0) close(dd->shm_fd);
        free(dd);
    } else {
        for (DeviceData *it = g_devices; it; it = it->next) {
            if (it->device == dev) { if (!it->DestroyDevice) RESOLVE(it, DestroyDevice); it->DestroyDevice(dev, alloc); break; }
        }
    }
}

/* ------------------------- dispatch ------------------------- */

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL clank_depth_GetDeviceProcAddr(VkDevice dev, const char *name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL clank_depth_GetInstanceProcAddr(VkInstance inst, const char *name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL clank_depth_GetPhysicalDeviceProcAddr(VkInstance inst, const char *name);

#define HOOK(fn) if (strcmp(name, "vk" #fn) == 0) return (PFN_vkVoidFunction)clank_depth_##fn;

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL clank_depth_GetDeviceProcAddr(VkDevice dev, const char *name)
{
    DeviceData *dd = find_device(dev);
    if (!dd) return NULL;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)clank_depth_GetDeviceProcAddr;
    HOOK(CreateImage) HOOK(DestroyImage)
    HOOK(CreateImageView) HOOK(DestroyImageView)
    HOOK(CreateFramebuffer) HOOK(DestroyFramebuffer)
    HOOK(CreateRenderPass) HOOK(DestroyRenderPass)
    HOOK(CreateCommandPool) HOOK(DestroyCommandPool)
    HOOK(AllocateCommandBuffers) HOOK(FreeCommandBuffers)
    HOOK(CmdBeginRenderPass) HOOK(CmdEndRenderPass)
    HOOK(CmdBeginRenderPass2) HOOK(CmdEndRenderPass2)
    HOOK(CmdBeginRendering) HOOK(CmdEndRendering)
    HOOK(CmdBeginRenderingKHR) HOOK(CmdEndRenderingKHR)
    HOOK(CreateSwapchainKHR) HOOK(GetSwapchainImagesKHR) HOOK(DestroySwapchainKHR)
    HOOK(DestroyDevice)
    return dd->next_gdpa(dev, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL clank_depth_GetInstanceProcAddr(VkInstance inst, const char *name)
{
    if (strcmp(name, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)clank_depth_GetInstanceProcAddr;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)clank_depth_GetDeviceProcAddr;
    if (strcmp(name, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)clank_depth_CreateInstance;
    if (strcmp(name, "vkEnumeratePhysicalDevices") == 0) return (PFN_vkVoidFunction)clank_depth_EnumeratePhysicalDevices;
    if (strcmp(name, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)clank_depth_CreateDevice;
    InstanceData *id = find_instance(inst);
    if (id) return id->next_gipa(inst, name);
    /* Global queries before instance creation. */
    return NULL;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL clank_depth_GetPhysicalDeviceProcAddr(VkInstance inst, const char *name)
{
    InstanceData *id = find_instance(inst);
    if (id && id->next_gpdpa) return id->next_gpdpa(inst, name);
    return NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL clank_depth_NegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface *p)
{
    if (p->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (p->loaderLayerInterfaceVersion > 2) p->loaderLayerInterfaceVersion = 2;
    p->pfnGetInstanceProcAddr = clank_depth_GetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = clank_depth_GetDeviceProcAddr;
    p->pfnGetPhysicalDeviceProcAddr = clank_depth_GetPhysicalDeviceProcAddr;
    return VK_SUCCESS;
}
