import { useUpload } from "@/hooks/use-upload";
import { cn } from "@/lib/utils";
import { uploadSchema } from "@/schema";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import axios from "axios";
import { Check, Image, Loader } from "lucide-react";
import { useState } from "react";
import { useDropzone } from "react-dropzone";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel } from "./ui/form";
import { RadioGroup, RadioGroupItem } from "./ui/radio-group";

const Uploader = () => {
    const [type, setType] = useState<"image" | "video">("image");
    const [preview, setPreview] = useState<string | null>(null);

    const { setType: setBlobType, setUploadUrl } = useUpload();
    const { getRootProps, getInputProps } = useDropzone({
        multiple: false,
        accept: {
            "image/*": [".jpeg", ".png", ".webp", ".jpg"],
            "video/*": [".mp4"],
        },
        onDrop: (acceptedFiles) => {
            const file = acceptedFiles[0];
            form.setValue("file", file);

            if (file.type.includes("image")) {
                setType("image");
                const fileReader = new FileReader();
                fileReader.readAsDataURL(file);
                fileReader.onload = () => {
                    setPreview(fileReader.result as string);
                };
            } else if (file.type.includes("video")) {
                setType("video");
                const objectUrl = URL.createObjectURL(file);
                setPreview(objectUrl);

                return () => URL.revokeObjectURL(objectUrl);
            }
        },
    });

    const form = useForm<z.infer<typeof uploadSchema>>({
        resolver: zodResolver(uploadSchema),
        defaultValues: {
            type: "image",
            model: "dcp",
        },
    });

    const { mutate, isPending, isSuccess } = useMutation({
        mutationKey: ["upload"],
        mutationFn: async (formData: FormData) => {
            const apiUrl = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");
            const response = await axios.post(`${apiUrl}/dehaze`, formData, { responseType: "blob" });
            return response;
        },
        onSuccess: (data) => {
            const blob = data.data;
            const imageData = URL.createObjectURL(blob);
            setUploadUrl(imageData);
        },
    });

    const onSubmit = (values: z.infer<typeof uploadSchema>) => {
        const formData = new FormData();
        formData.append("type", values.type);
        formData.append("model", values.model);
        formData.append("file", values.file);
        setBlobType(values.file.type.includes("image") ? "image" : "video");
        mutate(formData);
    };

    return (
        <Card className="h-fit md:flex-1">
            <CardHeader>
                <CardTitle>Upload Hazy Image/Video</CardTitle>
            </CardHeader>
            <CardContent>
                <Form {...form}>
                    <form onSubmit={form.handleSubmit(onSubmit)}>
                        <FormField
                            control={form.control}
                            name="type"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Type of file</FormLabel>
                                    <FormControl>
                                        <RadioGroup
                                            onValueChange={field.onChange}
                                            defaultValue={field.value}
                                            className="flex items-center gap-10"
                                        >
                                            <FormItem className="flex items-center space-x-3 space-y-0">
                                                <FormControl>
                                                    <RadioGroupItem value="image" />
                                                </FormControl>
                                                <FormLabel className="font-normal">Image</FormLabel>
                                            </FormItem>
                                            <FormItem className="flex items-center space-x-3 space-y-0">
                                                <FormControl>
                                                    <RadioGroupItem value="video" />
                                                </FormControl>
                                                <FormLabel className="font-normal">Video</FormLabel>
                                            </FormItem>
                                        </RadioGroup>
                                    </FormControl>
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="model"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Choose model</FormLabel>
                                    <FormControl>
                                        <RadioGroup
                                            onValueChange={field.onChange}
                                            defaultValue={field.value}
                                            className="flex items-center gap-10"
                                        >
                                            <FormItem className="flex items-center space-x-3 space-y-0">
                                                <FormControl>
                                                    <RadioGroupItem value="dcp" />
                                                </FormControl>
                                                <FormLabel className="font-normal">Dark Channel Prior</FormLabel>
                                            </FormItem>
                                            <FormItem className="flex items-center space-x-3 space-y-0">
                                                <FormControl>
                                                    <RadioGroupItem value="ffa" />
                                                </FormControl>
                                                <FormLabel className="font-normal">Feature Fusion Attention</FormLabel>
                                            </FormItem>
                                        </RadioGroup>
                                    </FormControl>
                                </FormItem>
                            )}
                        />
                        <div
                            {...getRootProps()}
                            className={cn("border-2 h-fit w-full my-4 border-dashed border-border rounded-lg", {
                                "border-red-500": !!form.formState.errors.file,
                            })}
                        >
                            {preview && (
                                <div className="h-full w-full p-2 rounded-lg">
                                    {type === "image" ? (
                                        <img
                                            src={preview}
                                            alt="Preview"
                                            className="aspect-video w-full object-cover rounded-lg"
                                        />
                                    ) : (
                                        <video
                                            src={preview}
                                            controls
                                            className="aspect-video w-full object-covert rounded-lg"
                                        />
                                    )}
                                </div>
                            )}
                            {!preview && (
                                <div className="h-64 w-full flex flex-col justify-center items-center gap-2">
                                    <Image className="size-5" />
                                    <h3>
                                        <span className="font-medium">Drag n Drop</span> or{" "}
                                        <span className="font-medium">Click to Upload</span>
                                    </h3>
                                </div>
                            )}
                            <input
                                {...getInputProps()}
                                type="file"
                                id="image"
                                accept="image/*,video/*"
                                className="hidden"
                            />
                        </div>
                        <p className="text-red-500 mb-2">{form.formState.errors.file?.message}</p>
                        <Button className="w-full gap-2">
                            {isPending && <Loader className="size-5 animate-spin" />}
                            {isSuccess && <Check className="size-5" />}
                            {isPending ? "Uploading..." : "Upload"}
                        </Button>
                    </form>
                </Form>
            </CardContent>
        </Card>
    );
};

export default Uploader;
